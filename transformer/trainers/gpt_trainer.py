import logging
import torch
from typing import Dict, List
from transformer.models.gpt import Gpt2
from transformer.trainers.learning_rate_lambda import LearningRateLambda
from transformer.data.gpt_data_loader import GptDataLoader
from transformer.trainers.interface import TrainerInterface
from transformer.losses.criterions import LanguageModelingCriterion, UnlikelihoodCriterion
from transformer.trainers.utils import ModelFilenameConstants, load_model_hyperparams, load_optimizer_hyperparams, load_state_dict, load_history, is_model_saved, is_optimizer_saved, is_history_saved


class GptTrainer(TrainerInterface):
    def __init__(self, temp_dir="./"):
        TrainerInterface.__init__(self, temp_dir=temp_dir)

    def get_init_params(self, **kwargs):
        required_parameters = ["timesteps", "vocab_size", "embedding_dict"]
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="required_parameters")
        default_parameters = [("pretrained_name_or_path", "skt/kogpt2-base-v2")]
        params = TrainerInterface.get_init_params(self=self, required_parameters=required_parameters, default_parameters=default_parameters, kwargs=kwargs)
        return params

    def create_model(self, **params):
        model = Gpt2(**params)
        return model

    def create_data_loader(self, **params):
        data_loader = GptDataLoader(**params)
        return data_loader

    def iteration(self, model, batch: List[Dict[str, torch.Tensor]], criterions, criterion_weights, optimizer, train: bool, amp: bool = False, scaler: torch.cuda.amp.GradScaler = None):
        inputs, outputs = batch

        with torch.cuda.amp.autocast(enabled=amp):
            # forward
            predictions = model.forward(inputs=inputs, outputs=outputs)
            _predictions = predictions["lm"]

            # backward
            loss_dict = dict()
            acc_dict = dict()

            loss = predictions["loss"]
            loss_dict["lm"] = loss.item()

            criterion = criterions["lm"]
            if model.is_log_prob: perplexity = torch.exp(loss)
            acc_dict["ppl"] = perplexity.item()
            _bleu_score = criterion.get_bleu_score(predictions=_predictions, targets=outputs["lm"])
            acc_dict["bleu"] = _bleu_score

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
        required_parameters = ["timesteps", "vocab_size", "pad_token_id"]
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="parameters")
        timesteps = kwargs.pop("timesteps")
        vocab_size = kwargs.pop("vocab_size")
        pad_token_id = kwargs.pop("pad_token_id")
        if "lm" in kwargs: lm = kwargs.pop("lm")
        else: lm = 1.0
        if "ul" in kwargs: ul = kwargs.pop("ul")
        else: ul = 0.0
        if "ngram" in kwargs: ngram = kwargs.pop("ngram")
        else: ngram = 4
        if "clipping" in kwargs: clipping = kwargs.pop("clipping")
        else: clipping = True
        if "is_log_probs" in kwargs: is_log_probs = kwargs.pop("is_log_probs")
        else: is_log_probs = True

        self.assert_equal(a=1.0, b=lm)
        criterions = dict()
        criterion_weights = dict()
        if lm > 0.0:
            criterions["lm"] = LanguageModelingCriterion(ignore_index=pad_token_id, ngram=ngram, clipping=clipping)
            criterion_weights["lm"] = lm
        if ul > 0.0:
            criterions["ul"] = UnlikelihoodCriterion(timesteps=timesteps, vocab_size=vocab_size, ignore_index=pad_token_id, ngram=ngram, is_log_probs=is_log_probs)
            criterion_weights["ul"] = ul
            self.unlikelyhood_loss_update = True
        return criterions, criterion_weights

    def get_optimizer(self, model, initial_learning_rate: float = 1e-4) -> torch.optim.Optimizer:
        return TrainerInterface.get_optimizer(self=self, optimizer="adam_w", model=model, initial_learning_rate=initial_learning_rate)

    def get_data_loader_params(self, **kwargs):
        common_required_parameters = TrainerInterface.get_data_loader_params(self=self, **kwargs)
        custom_required_parameters = ["timesteps", "embedding_dict", "src_sep_tokens"]
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
        # TODO
        if not path.endswith("/"): path += "/"
        output = dict()
        # load model
        if is_model_saved(path=path, save_hyperparms=True):
            model_hyperparams = load_model_hyperparams(path=path + ModelFilenameConstants.MODEL_HYPERPARAMS_FILENAME)
            model_hyperparams = {k: v for k, v in model_hyperparams.items() if k in Bert.__init__.__code__.co_varnames}
            model = Gpt2(**model_hyperparams)
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

