from collections import Counter
import logging
import torch
import numpy as np
from typing import Dict, List
from transformer.trainers.interface import TrainerInterface
from transformer.trainers.learning_rate_lambda import LearningRateLambda
from transformer.losses.criterions import LanguageModelingCriterion, UnlikelihoodCriterion
from transformer.trainers.utils import ModelFilenameConstants, load_model_hyperparams, load_optimizer_hyperparams, load_state_dict, load_history, is_model_saved, is_optimizer_saved, is_history_saved
from transformer.models.transformer import Transformer
from transformer.data.transformer_data_loader import TransformerDataLoader


class TransformerTrainer(TrainerInterface):
    unlikelyhood_loss_update = False
    special_token_ids = None
    target_prev_token_distribution = None
    predicted_prev_token_distribution = None
    most_different_tokens = None

    def __init__(self, temp_dir="./"):
        TrainerInterface.__init__(self, temp_dir=temp_dir)

    def get_init_params(self, **kwargs):
        required_parameters = ["src_timesteps", "src_vocab_size", "src_pad_token_id", "tgt_timesteps", "tgt_vocab_size", "tgt_pad_token_id",
                               "embedding_dict", "d_model", "d_ff", "num_heads", "num_encoder_layers", "num_decoder_layers", "shared_embedding"]
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="required_parameters")
        default_parameters = [("dropout", 0.1), ("pwff_activation", "gelu"), ("linear_activation", "gelu"), ("bias", True), ("layer_norm_epsilon", 1e-5), ("initialization", "normal")]
        params = TrainerInterface.get_init_params(self=self, required_parameters=required_parameters, default_parameters=default_parameters, kwargs=kwargs)
        return params

    def create_model(self, **params):
        model = Transformer(**params)
        return model

    def create_data_loader(self, **params):
        data_loader = TransformerDataLoader(**params)
        return data_loader

    def iteration(self, model, batch: List[Dict[str, torch.Tensor]],
                  criterions, criterion_weights, optimizer, train: bool, amp: bool = False, scaler: torch.cuda.amp.GradScaler = None):
        loss = None
        src_inputs, tgt_inputs, tgt_outputs = batch
        with torch.cuda.amp.autocast(enabled=amp):
            # forward
            predictions = model.forward(src_inputs=src_inputs, tgt_inputs=tgt_inputs)

            # backward
            loss_dict = dict()
            acc_dict = dict()
            for name, criterion in criterions.items():
                if name == "ul": continue
                criterion_weight = criterion_weights[name]
                _predictions = predictions[name]
                _targets = tgt_outputs[name]

                _loss = criterion.get_loss(predictions=_predictions, targets=_targets)
                _loss = criterion_weight * _loss
                _accuracy = criterion.get_accuracy(predictions=_predictions, targets=_targets.long())
                if loss is None: loss = _loss
                else: loss += _loss
                loss_dict[name] = _loss.item()
                acc_dict[name] = _accuracy.item()
                if name == "lm":
                    # _perplexity = criterion.get_perplexity(predictions=_predictions, targets=_targets.long())
                    _perplexity = _loss
                    if criterion.is_log_prob: _perplexity = torch.exp(_loss)
                    acc_dict["ppl"] = _perplexity.item()
                    _bleu_score = criterion.get_bleu_score(predictions=_predictions, targets=_targets)
                    acc_dict["bleu"] = _bleu_score

                if name == "lm" and "ul" in criterions:
                    criterion = criterions["ul"]

                    # change _targets to ulk_tagets
                    prediction_token_ids = torch.argmax(_predictions, axis=-1)
                    pad_mask = (src_inputs["token"] != criterion.ignore_index)
                    prediction_token_ids = prediction_token_ids * pad_mask

                    # ulk_targets: (batch_size, timsteps, timesteps)
                    ulk_targets = None
                    if self.target_prev_token_distribution is not None:
                        ulk_targets = self.get_ulk_targets(prediction_token_ids=prediction_token_ids, ignore_index=criterion.ignore_index, ngram=criterion.ngram)
                        # update predicted_prev_token_distribution
                        prediction_token_ids = self.convert_to_numpy(tensor=prediction_token_ids)
                        self.update_prev_token_distribution(prediction_token_ids=prediction_token_ids, ngram=criterion.ngram)
                        self.update_most_different_tokens(ngram=criterion.ngram)
                    else:
                        ulk_targets = self.get_default_ulk_targets(prediction_token_ids=prediction_token_ids, timesteps=criterion.timesteps, ngram=criterion.ngram)

                    criterion_weight = criterion_weights["ul"]
                    _loss = criterion.get_loss(predictions=_predictions, targets=ulk_targets.long())
                    _loss = criterion_weight * _loss
                    _accuracy = criterion.get_accuracy(predictions=_predictions, targets=_targets)
                    if loss is None: loss = _loss
                    else: loss += _loss
                    loss_dict["ul"] = _loss.item()
                    acc_dict["ul"] = _accuracy.item()

        # optimize
        if train:
            if self.lr_update:
                lr = self.lr_lambda(current_step=self.step)
                optimizer = self.update_optimizer_lr(optimizer=optimizer, lr=lr)
            if amp and isinstance(scaler, torch.cuda.amp.GradScaler):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            self.step += 1
        return loss_dict, acc_dict

    def set_lr_update(self, initial_learning_rate, num_warmup_steps):
        _lambda = None
        if num_warmup_steps > 0:
            _lambda = "transformer_lambda"
            self.lr_update = True
            self.lr_lambda = LearningRateLambda(initial_learning_rate=initial_learning_rate, num_warmup_steps=num_warmup_steps).transformer_lambda
        message = "LearningRate schedule has been set to '{_lambda}'".format(_lambda=_lambda)
        print(message)

    def get_criterions(self, **kwargs):
        '''
        :param lm: default 1.0
        :param ul: unlikelyhood_alpha, default 0.5
        :return:
        '''
        required_parameters = ["tgt_timesteps", "tgt_vocab_size", "tgt_pad_token_id"]
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="parameters")
        tgt_timesteps = kwargs.pop("tgt_timesteps")
        tgt_vocab_size = kwargs.pop("tgt_vocab_size")
        tgt_pad_token_id = kwargs.pop("tgt_pad_token_id")
        if "lm" in kwargs: lm = kwargs.pop("lm")
        else: lm = 1.0
        if "ul" in kwargs: ul = kwargs.pop("ul")
        else: ul = 0.0
        if "ngram" in kwargs: ngram = kwargs.pop("ngram")
        else: ngram = 4
        if "clipping" in kwargs: clipping = kwargs.pop("clipping")
        else: clipping = 5
        if "is_log_probs" in kwargs: is_log_probs = kwargs.pop("is_log_probs")
        else: is_log_probs = True

        criterions = dict()
        criterion_weights = dict()
        if lm > 0.0:
            criterions["lm"] = LanguageModelingCriterion(ignore_index=tgt_pad_token_id, ngram=ngram, clipping=clipping)
            criterion_weights["lm"] = lm
        if ul > 0.0:
            criterions["ul"] = UnlikelihoodCriterion(timesteps=tgt_timesteps, vocab_size=tgt_vocab_size, ignore_index=tgt_pad_token_id, ngram=ngram, is_log_probs=is_log_probs)
            criterion_weights["ul"] = ul
            self.unlikelyhood_loss_update = True
        return criterions, criterion_weights

    def get_optimizer(self, model, initial_learning_rate: float = 1e-4, beta_1: float = 0.9, beta_2: float = 0.98, optimizer_epsilon: float = 1e-5) -> torch.optim.Optimizer:
        return TrainerInterface.get_optimizer(self=self, optimizer="transformer", model=model, initial_learning_rate=initial_learning_rate, beta_1=beta_1, beta_2=beta_2, optimizer_epsilon=optimizer_epsilon)

    def get_data_loader_params(self, **kwargs):
        common_required_parameters = TrainerInterface.get_data_loader_params(self=self, **kwargs)
        custom_required_parameters = ["src_timesteps", "tgt_timesteps", "embedding_dict", "src_sep_tokens"]
        self.assert_contain_elements(required=custom_required_parameters, target=kwargs, name="custom_required_parameters")
        required_parameters = common_required_parameters + custom_required_parameters
        default_parameters = [("num_workers", self.num_workers), ("pin_memory", self.pin_memory), ("approach", "ignore")]
        params = dict()
        for k, v in default_parameters:
            if k in kwargs: v = kwargs.pop(k)
            params[k] = v
        for k, v in kwargs.items():
            if k not in required_parameters: continue
            params[k] = v
        return params

    def load(self, path):
        if not path.endswith("/"): path += "/"
        output = dict()
        # load model
        if is_model_saved(path=path, save_hyperparms=True):
            model_hyperparams = load_model_hyperparams(path=path + ModelFilenameConstants.MODEL_HYPERPARAMS_FILENAME)
            model_hyperparams = {k: v for k, v in model_hyperparams.items() if k in Transformer.__init__.__code__.co_varnames}
            model = Transformer(**model_hyperparams)
            model = load_state_dict(object=model, path=path + ModelFilenameConstants.MODEL_STATE_DICT_FILENAME)
            output["model"] = model

        # load optimizer
        if is_optimizer_saved(path=path, save_hyperparms=True):
            optimizer_hyperparams = load_optimizer_hyperparams(path=path + ModelFilenameConstants.OPTIMIZER_HYPERPARAMS_FILENAME)
            optimizer_hyperparams = {k: v for k, v in optimizer_hyperparams.items() if k in self.get_optimizer.__code__.co_varnames}
            optimizer = self.get_optimizer(model=model, **optimizer_hyperparams)
            optimizer = load_state_dict(path=path + ModelFilenameConstants.OPTIMIZER_STATE_DICT_FILENAME, object=optimizer)
            output["optimizer"] = optimizer

        # load history
        if is_history_saved(path=path):
            history = load_history(path=path + ModelFilenameConstants.HISTORY_FILENAME)
            output["history"] = history

        message = "Loaded '{contents}' from {path}".format(contents=output.keys(), path=path)
        print(message)
        logging.info(message)
        return output

    def set_prev_token_distribution(self, prev_token_distribution, special_token_ids):
        self.special_token_ids = special_token_ids
        self.target_prev_token_distribution = prev_token_distribution
        self.predicted_prev_token_distribution = dict()
        self.most_different_tokens = dict()

    def init_prev_token_distribution(self):
        self.predicted_prev_token_distribution = dict()
        self.most_different_tokens = dict()

    def update_prev_token_distribution(self, prediction_token_ids, ngram):
        for batch_idx, prediction_token_ids_row in enumerate(prediction_token_ids):
            for target_idx in range(0, len(prediction_token_ids_row)):
                target_token = prediction_token_ids_row[target_idx]
                if target_token in self.special_token_ids: continue
                begin_idx = min(0, (target_idx - ngram - 1))
                tokens_to_update = prediction_token_ids_row[begin_idx:target_idx]

                if target_token not in self.predicted_prev_token_distribution:
                    self.predicted_prev_token_distribution[target_token] = Counter()
                self.predicted_prev_token_distribution[target_token].update(tokens_to_update)

    def update_most_different_tokens(self, ngram):
        most_different_tokens = dict()
        # get most different tokens between target_prev_token_distribution and predicted_prev_token_distribution
        for token_id, _predicted_counter in self.predicted_prev_token_distribution.items():
            if token_id in self.special_token_ids: continue
            token_ids = list(_predicted_counter.keys())
            predicted_frequency = np.array(list(_predicted_counter.values()))
            predicted_distribution = predicted_frequency / np.sum(predicted_frequency)
            predicted_counter = Counter(dict(zip(token_ids, predicted_distribution)))
            target_counter = self.target_prev_token_distribution[token_id]
            difference_counter = predicted_counter - target_counter
            ulk_token_ids = [_token_id for _token_id, difference in difference_counter.most_common(ngram)]
            most_different_tokens[token_id] = ulk_token_ids
        self.most_different_tokens = most_different_tokens

    def get_default_ulk_targets(self, prediction_token_ids: torch.Tensor, timesteps: int, ngram: int) -> torch.Tensor:
        # prediction_token_ids: (batch_size, timsteps)
        # ulk_targets: (batch_size, timsteps, timesteps)
        prediction_token_ids_expanded = prediction_token_ids.unsqueeze(1).expand(-1, timesteps, timesteps)
        mask_upper_than_ngram = prediction_token_ids_expanded.tril(-1).bool().long()
        mask_less_than_ngram = prediction_token_ids_expanded.triu(-ngram).bool().long()
        ulk_targets = prediction_token_ids_expanded * mask_upper_than_ngram * mask_less_than_ngram
        return ulk_targets

    def get_ulk_targets(self, prediction_token_ids: torch.Tensor, ignore_index: int, ngram: int) -> torch.Tensor:
        # prediction_token_ids: (batch_size, timsteps)
        # ulk_targets: (batch_size, timsteps, ngram)
        ulk_targets = []
        empty_row = [ignore_index] * ngram
        for row in prediction_token_ids:
            ulk_token_row = []
            for token_id in row:
                _row = empty_row # set prev_token_ids of tokens previously not observed to empty_row
                if token_id.item() in self.most_different_tokens:
                    _row = self.most_different_tokens[token_id.item()]
                    if len(_row) < ngram:
                        _row += [ignore_index] * (ngram - len(_row))
                ulk_token_row.append(_row)
            ulk_targets.append(ulk_token_row)
        ulk_targets = self.convert_to_tensor(data=ulk_targets, device=prediction_token_ids.device)
        return ulk_targets