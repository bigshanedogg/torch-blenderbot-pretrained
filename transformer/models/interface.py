import re
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import torch
from KoBERTScore import BERTScore
from transformer.assertions.interface import Assertion
from transformer.assertions.object_assertion import ModelAssertion
from transformer.models.utils import save_model, save_optimizer, save_tokenizer, load_state_dict, load_tokenizer
from transformer.models.utils import compute_bleu, compute_meteor, compute_rouge, compute_hits, compute_semantic_score
from transformer.utils.common import get_now_str, convert_to_tensor, convert_to_numpy, clean_text, init_path
from datasets import load_metric

class ModelInterface(ModelAssertion):
    __name__ = None

    def __init__(self, temp_dir="./"):
        if temp_dir.endswith("/"): temp_dir = temp_dir[:-1]
        self.temp_dir = "./{now}/".format(model_path=temp_dir, now=get_now_str(str_format="%Y%m%d_%H%M%S"))
        self.verbose_template = "{mode} ({device}) [{idx:^3d}/{num_iters:^3d}]:"
        print("'temp_dir' has been set to '{temp_dir}' to save model while training".format(temp_dir=self.temp_dir))
        self.metrics = self._get_default_metrics()
        self.default_generate_params = self._get_default_generate_params()

    def get_optimizer(self, optimizer: str, **kwargs):
        self.assert_isin_optimizers(optimizer=optimizer)
        if optimizer == "adam":
            required_parameters = ["initial_learning_rate", "beta_1", "beta_2", "optimizer_epsilon"]
            self.assert_contain_elements(required=required_parameters, target=kwargs, name="parameters")
            initial_learning_rate = kwargs["initial_learning_rate"]
            beta_1 = kwargs["beta_1"]
            beta_2 = kwargs["beta_2"]
            optimizer_epsilon = kwargs["optimizer_epsilon"]
            optimizer = torch.optim.Adam(self.parameters(), lr=initial_learning_rate, betas=(beta_1, beta_2), eps=optimizer_epsilon)
        elif optimizer == "adam_w":
            required_parameters = ["initial_learning_rate"]
            self.assert_contain_elements(required=required_parameters, target=kwargs, name="parameters")
            initial_learning_rate = kwargs["initial_learning_rate"]
            optimizer = torch.optim.AdamW(self.parameters(), lr=initial_learning_rate)

        if self.assert_isinstance_optimizer(optimizer=optimizer):
            optimizer.__dict__.update(kwargs)
        return optimizer

    def fit(self, epoch: int, optimizer, device, train_data_loader, val_data_loader, save_per_epoch: int = -1, verbose_per_epoch: int = -1, verbose_per_batch: int = -1):
        init_path(self.temp_dir, True)
        train_history = TrainHistory()
        val_history = TrainHistory()
        for _epoch in range(1, epoch + 1):
            # train
            epoch_train_history = self.iteration_epoch(data_loader=train_data_loader, optimizer=optimizer, device=device, train=True, verbose_per_batch=verbose_per_batch)
            epoch_train_history_str = self.verbose_template.format(mode="Epoch_train", device=device, idx=_epoch, num_iters=len(train_data_loader)) + str(epoch_train_history)
            train_history += epoch_train_history
            if verbose_per_epoch > 0 and _epoch % verbose_per_epoch == 0: print(epoch_train_history_str)
            if save_per_epoch > 0 and _epoch % save_per_epoch == 0:
                self.save(path=self.temp_dir + "epoch_{}/".format(_epoch), optimizer=optimizer, tokenizer=None)

            # val
            if val_data_loader is not None:
                epoch_val_history = self.iteration_epoch(data_loader=val_data_loader, optimizer=optimizer, device=device, train=False, verbose_per_batch=-1)
                epoch_val_history_str = self.verbose_template.format(mode="Epoch_val", device=device, idx=_epoch, num_iters=len(val_data_loader)) + str(epoch_val_history)
                print(epoch_val_history_str)
                if verbose_per_epoch > 0 and _epoch % verbose_per_epoch == 0: val_history += epoch_val_history

            with open(self.temp_dir + "log.txt", "a", encoding="utf-8") as fp:
                fp.write(epoch_train_history_str + "\n")
                fp.write(epoch_val_history_str + "\n")

    def iteration_epoch(self, data_loader, optimizer, device, train: bool = False, verbose_per_batch: int = -1):
        epoch_history = TrainHistory()
        batch_history = TrainHistory()

        desc = "train" if train else "val"
        if train: self.train()
        else: self.eval()
        data_iter = tqdm(data_loader, initial=0, total=len(data_loader), desc=desc)
        for batch_idx, batch in enumerate(data_iter):
            batch_idx += 1
            outputs = self.iteration_batch(batch=batch, device=device)
            loss = outputs["loss"]

            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            loss_dict, acc_dict = self._get_history_dict(outputs=outputs)
            batch_history.update(loss_dict=loss_dict, acc_dict=acc_dict, lr=optimizer.param_groups[0]["lr"])
            if train and verbose_per_batch > 0 and (batch_idx + 1) % 500 == 0:
                batch_history_str = self.verbose_template.format(mode="Batch_{}".format(desc), device=device, idx=batch_idx, num_iters=len(data_loader)) + str(batch_history)
                print(batch_history_str)
                epoch_history += batch_history
                batch_history = TrainHistory()

        if batch_history.iteration > 0: epoch_history += batch_history
        return epoch_history

    def iteration_batch(self):
        self.assert_implemented(method_name="iteration_batch")

    def compute_scores(self, metrics:List[str], tokenizer, data_loader, device, **kwargs):
        scores = dict()
        bleu_ngrams = [3, 4]
        rouge_types = ["1", "2", "L"]
        hits_k = [1, 2, 5, 10]
        name_or_path = "beomi/kcbert-base"
        for metric in metrics:
            self.assert_isin_metrics(metric=metric)
        if "bleu" in metrics:
            if "bleu_ngrams" in kwargs: bleu_ngrams = kwargs["bleu_ngrams"]
            else: kwargs["bleu_ngrams"] = bleu_ngrams
        if "rouge" in metrics:
            if "rouge_types" in kwargs: rouge_types = kwargs["rouge_types"]
            else: kwargs["rouge_nums"] = rouge_types
        if "hits" in metrics:
            if "hits_k" in kwargs: hits_k = kwargs["hits_k"]
            else: kwargs["hits_k"] = hits_k
        if "semantic_score" in metrics:
            if "name_or_path" in kwargs: name_or_path = kwargs.pop("name_or_path")
            if "semantic_score" not in self.metrics:
                self.metrics["semantic_score"] = BERTScore(model_name_or_path=name_or_path, best_layer=-1, device=device)

        _scores = self._compute_scores(metrics=metrics, tokenizer=tokenizer, data_loader=data_loader, device=device, **kwargs)

        if "bleu" in metrics:
            _bleu_scores = _scores["bleu"]["precisions"]
            for ngram in bleu_ngrams:
                name = "BLEU-{n}".format(n=ngram)
                score = _bleu_scores[ngram-1]
                score = round(score, 4)
                scores[name] = score
        if "meteor" in metrics:
            score = _scores["meteor"]["meteor"]
            score = round(score, 4)
            scores["METEOR"] = score
        if "rouge" in metrics:
            for r in rouge_types:
                key = "rouge{r}".format(r=r)
                if key in _scores["rouge"]:
                    name = "ROUGE-{r}".format(r=r)
                    score = _scores["rouge"][key]
                    score = score.mid.fmeasure
                    score = round(score, 4)
                    scores[name] = score
        if "hits" in metrics:
            for k, score in zip(hits_k, _scores["hits"]):
                name = "HITS@{k}".format(k=k)
                score = round(score, 4)
                scores[name] = score
        if "semantic_score" in metrics:
            name = "BERTScore".format(name_or_path=name_or_path)
            score = _scores["semantic_score"]
            score = round(score, 4)
            scores[name] = score
        return scores

    def _compute_scores(self, metrics, tokenizer, data_loader, device, **kwargs):
        self.assert_implemented(method_name="_compute_scores")

    def _compute_generative_scores(self, metrics, tokenizer, data_loader, device, **kwargs):
        required_parameters = ["timesteps", "decoding_method"]
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="parameters")
        timesteps = kwargs.pop("timesteps")
        decoding_method = kwargs.pop("decoding_method")
        self.assert_isin_decoding_methods(method=decoding_method)

        predictions, references = self.get_metric_inputs(tokenizer=tokenizer, data_loader=data_loader, device=device, timesteps=timesteps, decoding_method=decoding_method, **kwargs)

        scores = dict()
        if "bleu" in metrics:
            scores["bleu"] = compute_bleu(metric=self.metrics["bleu"], tokenizer=tokenizer, predictions=predictions, references=references)
        if "meteor" in metrics:
            scores["meteor"] = compute_meteor(metric=self.metrics["meteor"], tokenizer=tokenizer, predictions=predictions, references=references)
        if "rouge" in metrics:
            scores["rouge"] = compute_rouge(metric=self.metrics["rouge"], tokenizer=tokenizer, predictions=predictions, references=references)
        if "semantic_score" in metrics:
            scores["semantic_score"] = compute_semantic_score(metric=self.metrics["semantic_score"], tokenizer=tokenizer, predictions=predictions, references=references)
        return scores

    def _compute_discrimitive_scores(self, metrics, tokenizer, data_loader, device, **kwargs):
        required_parameters = ["hits_k"]
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="parameters")
        hits_k = kwargs.pop("hits_k")

        # predictions: List[List[str]]
        # references: List[str]
        predictions, references = self.get_metric_inputs(tokenizer=tokenizer, data_loader=data_loader, device=device, hits_k=hits_k, **kwargs)

        scores = dict()
        if "hits" in metrics:
            # hits requires predictions: List[List[any]], references: List[str]
            scores["hits"] = compute_hits(predictions=predictions, references=references, k=hits_k)
        if "semantic_score" in metrics:
            # BERTScore requires predictions: List[str], references: List[str]
            _predictions = [prediction[0] for prediction in predictions]
            scores["semantic_score"] = compute_semantic_score(metric=self.metrics["semantic_score"], tokenizer=tokenizer, predictions=_predictions, references=references)
        return scores

    def get_metric_inputs(self):
        self.assert_implemented(method_name="get_metric_inputs")

    def _get_default_metrics(self):
        metrics = {
            "bleu": load_metric("bleu"),
            "meteor": load_metric("meteor"),
            "rouge": load_metric("rouge")
        }
        return metrics

    def _get_default_generate_params(self):
        '''
        must be identical to params of microservices/models to compare strictly
        :return:
        '''
        params = dict()
        params["greedy"] = {
            "min_length": 10,
            "repetition_penalty": 2.0,
            "no_repeat_ngram_size": 3
        }
        params["beam_search"] = {
            "min_length": 10,
            "repetition_penalty": 2.0,
            "no_repeat_ngram_size": 3,
            "num_beams": 5,
            "early_stopping": True,
            "num_return_sequences": 1
        }
        params["top_k_sampling"] = {
            "min_length": 10,
            "repetition_penalty": 2.0,
            "no_repeat_ngram_size": 3,
            "top_k": 40,
            "top_p": 0.95,
            "num_return_sequences": 1,
        }
        return params

    def _get_history_dict(self, outputs: Dict[str, torch.Tensor]):
        loss_dict = dict()
        acc_dict = dict()
        for k, v in outputs.items():
            if k.endswith("_loss"):
                k = k[:-5]
                loss_dict[k] = v
            elif k.endswith("_acc"):
                k = k[:-4]
                acc_dict[k] = v
            elif k in ["ppl", "bleu", "rouge"]:
                acc_dict[k] = v
            # elif k == "loss":
            #     loss_dict["total"] = v
        return loss_dict, acc_dict

    def save(self, path, optimizer, tokenizer, history=None):
        path = init_path(path, reset=True)

        save_model(path=path, model=self, ddp=False)
        if optimizer is not None:
            save_optimizer(path=path, optimizer=optimizer)
        if tokenizer is not None:
            save_tokenizer(path=path, tokenizer=tokenizer)
        if history is not None:
            # save_history(path=path, history=history)
            pass
        message = "Saved into {path}".format(path=path)
        print(message)
        return path

    def load(self, path: str, model_type: str):
        self.assert_is_valid_path(path=path)
        output = dict()
        self = load_state_dict(object=self, path=path)
        optimizer = self.get_optimizer()
        optimizer = load_state_dict(object=optimizer, path=path)
        tokenizer = load_tokenizer(path=path, model_type=model_type)
        output["optimizer"] = optimizer
        output["tokenizer"] = tokenizer
        return output

    def extract_candidates(self, device, dataset, additional_responses: List[str], concat_candidates: bool = True, verbose: bool = True):
        candidates = []
        candidate_embeds = []

        _candidate_inputs = []
        if dataset is not None:
            extract_iter = dataset.raw_data
            if verbose: extract_iter = tqdm(dataset.raw_data, initial=0, total=len(dataset.raw_data), desc="Extracting responses")
            for row in extract_iter:
                utterances = row["utterances"]
                speaker_ids = row["speaker_ids"]
                conditions = None
                if "conditions" in row:
                    conditions = row["conditions"]
                _, candidate_input = dataset.make_inputs(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
                _candidate_input_utterances = candidate_input["candidate"]
                if concat_candidates: _candidate_input_utterances = [" ".join(candidate_input["candidate"])]
                candidate_input_utterances = []
                for _candidate_input_utterance in _candidate_input_utterances:
                    _candidate_input_utterance = clean_text(_candidate_input_utterance)
                    if len(_candidate_input_utterance) < 5 or len(_candidate_input_utterance.split()) < 3: continue
                    candidate_input_utterances.append(_candidate_input_utterance)
                _candidate_inputs += candidate_input_utterances

        if additional_responses is not None:
            for additional_response in additional_responses:
                _candidate_input = clean_text(text=additional_response)
                _candidate_inputs.append(_candidate_input)

        _candidate_inputs = list(set(_candidate_inputs))
        embedding_iter = _candidate_inputs
        if verbose: embedding_iter = tqdm(_candidate_inputs, initial=0, total=len(_candidate_inputs), desc="Embedding responses")
        for candidate_input in embedding_iter:
            _candidate_input = {"candidate":[candidate_input], "speaker_ids":[0]}
            right_encoded = dataset.encode_right_row(candidate=_candidate_input)
            if right_encoded is None: continue

            _inputs = dict()
            _inputs["right_input_ids"] = right_encoded[0]
            _inputs["right_token_type_ids"] = right_encoded[1]
            _inputs["right_attention_mask"] = right_encoded[2]
            _inputs = {k: convert_to_tensor([v], device=device) for k, v in _inputs.items()}

            right_encoder_output = self.forward_right(inputs=_inputs)
            candidate_embed = right_encoder_output["right_encoder_logits"][0]
            candidate_embed = convert_to_numpy(tensor=candidate_embed)
            candidate_embeds.append(candidate_embed)
            candidates.append(candidate_input)
        candidate_embeds = np.array(candidate_embeds)

        self.assert_equal_length(a=candidates, b=candidate_embeds)
        print("Total {} candidates has been extracted".format(len(candidates)))
        return candidates, candidate_embeds

class TrainHistory(Assertion):
    iteration = 0
    begin_time = None
    end_time = None
    train_time = 0
    loss_dict = dict()
    acc_dict = dict()
    lr = []

    def __init__(self, underflow: float = 1e-7):
        # elements
        self.iteration = 0
        self.begin_time = datetime.now()
        self.loss_dict = dict()
        self.acc_dict = dict()
        self.lr = []
        # constants
        self.underflow = underflow

    def update(self, loss_dict: dict, acc_dict: dict, lr:float):
        self.iteration += 1
        self._add_lr(value=lr)
        for k,v in loss_dict.items():
            self._add_loss(name=k, value=v)
        for k,v in acc_dict.items():
            self._add_acc(name=k, value=v)

    def _add_lr(self, value):
        self.lr.append(value)

    def _add_loss(self, name, value):
        if name not in self.loss_dict: self.loss_dict[name] = []
        self.loss_dict[name].append(float(value))

    def _add_acc(self, name, value):
        if name not in self.acc_dict: self.acc_dict[name] = []
        self.acc_dict[name].append(float(value))

    def __add__(self, other):
        # assert
        self.assert_isintance(obj=other, data_type=TrainHistory)
        # merge
        self.iteration += other.iteration
        for name in other.loss_dict.keys():
            if name not in self.loss_dict: self.loss_dict[name] = []
            self.loss_dict[name] += other.loss_dict[name]
        for name in other.acc_dict.keys():
            if name not in self.acc_dict: self.acc_dict[name] = []
            self.acc_dict[name] += other.acc_dict[name]

        self.lr += other.lr
        return self

    def __repr__(self):
        repr = " (loss) {loss_repr} | (acc) {acc_repr} | train_time: {train_time:.1f}s, last_lr: {last_lr: .10f}"
        loss_template = "{key}: {value:.3e}, "
        acc_template = "{key}: {value:.3e}, "

        self.freeze()
        loss_repr = ""
        acc_repr = ""
        for name in self.loss_dict.keys():
            loss_repr += loss_template.format(key=name, value=(sum(self.loss_dict[name]) / (len(self.loss_dict[name]) + self.underflow)))
        for name in self.acc_dict.keys():
            acc_repr += acc_template.format(key=name, value=(sum(self.acc_dict[name]) / (len(self.acc_dict[name]) + self.underflow)))
        last_lr = -1
        if len(self.lr) > 0: last_lr = self.lr[-1]
        return repr.format(loss_repr=loss_repr, acc_repr=acc_repr, train_time=self.train_time, last_lr=last_lr)

    def freeze(self):
        # freeze train_time
        self.end_time = datetime.now()
        train_time = self.end_time - self.begin_time
        self.train_time = train_time.seconds

    def to_dict(self):
        self.freeze()
        output = OrderedDict()
        output["iteration"] = self.iteration
        output["begin_time"] = self.begin_time
        output["end_time"] = self.end_time
        output["train_time"] = self.train_time
        output["loss"] = self.loss_dict
        output["acc"] = self.acc_dict
        output["lr"] = self.lr
        return output