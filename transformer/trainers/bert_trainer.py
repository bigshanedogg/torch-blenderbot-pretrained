import logging
import torch
from typing import Dict, List
from transformer.models.bert import Bert
from transformer.trainers.learning_rate_lambda import LearningRateLambda
from transformer.data.bert_data_loader import BertDataLoader
from transformer.trainers.interface import TrainerInterface
from transformer.losses.criterions import LanguageModelingCriterion, LogLikelyhoodCriterion
from transformer.trainers.utils import ModelFilenameConstants, load_model_hyperparams, load_optimizer_hyperparams, load_state_dict, load_history, is_model_saved, is_optimizer_saved, is_history_saved


class BertTrainer(TrainerInterface):
    def __init__(self, temp_dir="./"):
        TrainerInterface.__init__(self, temp_dir=temp_dir)

    def get_init_params(self, **kwargs):
        required_parameters = ["timesteps", "vocab_size", "embedding_dict", "d_model", "d_ff", "num_heads", "num_layers", "shared_embedding", "pad_token_id"]
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="required_parameters")
        default_parameters = [("dropout", 0.1), ("pwff_activation", "gelu"), ("linear_activation", "gelu"), ("bias", True), ("layer_norm_epsilon", 1e-5), ("initialization", "normal")]
        params = TrainerInterface.get_init_params(self=self, required_parameters=required_parameters, default_parameters=default_parameters, kwargs=kwargs)
        return params

    def create_model(self, **params):
        model = Bert(**params)
        return model

    def create_data_loader(self, **params):
        data_loader = BertDataLoader(**params)
        return data_loader

    def iteration(self, model, batch: List[Dict[str, torch.Tensor]], criterions, criterion_weights, optimizer, train: bool, amp: bool = False, scaler: torch.cuda.amp.GradScaler = None):
        loss = None
        inputs, outputs = batch
        with torch.cuda.amp.autocast(enabled=amp):
            # forward
            predictions = model.forward(inputs=inputs)

            # backward
            loss_dict = dict()
            acc_dict = dict()
            for name, criterion in criterions.items():
                criterion_weight = criterion_weights[name]
                _predictions = predictions[name]
                _targets = outputs[name]
                _loss = criterion.get_loss(predictions=_predictions, targets=_targets)
                _loss = criterion_weight * _loss
                _accuracy = criterion.get_accuracy(predictions=_predictions, targets=_targets)
                if loss is None: loss = _loss
                else: loss += _loss

                loss_dict[name] = _loss.item()
                acc_dict[name] = _accuracy.item()
                if name == "mlm":
                    _perplexity = criterion.get_perplexity(predictions=_predictions, targets=_targets.long())
                    acc_dict["ppl"] = _perplexity.item()

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
        required_parameters = ["pad_token_id"]
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="parameters")
        pad_token_id = kwargs.pop("pad_token_id")
        if "mlm" in kwargs: mlm = kwargs.pop("mlm")
        else: mlm = 0.5
        if "nsp" in kwargs: nsp = kwargs.pop("nsp")
        else: nsp = 0.5
        if "sop" in kwargs: sop = kwargs.pop("sop")
        else: sop = 0.0
        if "ngram" in kwargs: ngram = kwargs.pop("ngram")
        else: ngram = 4
        if "clipping" in kwargs: clipping = kwargs.pop("clipping")
        else: clipping = True

        self.assert_equal(a=1.0, b=(mlm + nsp + sop))
        criterions = dict()
        criterion_weights = dict()
        if mlm > 0.0:
            criterions["mlm"] = LanguageModelingCriterion(ignore_index=pad_token_id, ngram=ngram, clipping=clipping)
            criterion_weights["mlm"] = mlm
        if nsp > 0.0:
            criterions["nsp"] = LogLikelyhoodCriterion()
            criterion_weights["nsp"] = nsp
        if sop > 0.0:
            criterions["sop"] = LogLikelyhoodCriterion() # TODO
            criterion_weights["sop"] = sop
        return criterions, criterion_weights

    def get_optimizer(self, model, initial_learning_rate: float = 1e-4, beta_1: float = 0.9, beta_2: float = 0.98, optimizer_epsilon: float = 1e-5) -> torch.optim.Optimizer:
        return TrainerInterface.get_optimizer(self=self, optimizer="transformer", model=model, initial_learning_rate=initial_learning_rate, beta_1=beta_1, beta_2=beta_2, optimizer_epsilon=optimizer_epsilon)

    def get_data_loader_params(self, **kwargs):
        common_required_parameters = TrainerInterface.get_data_loader_params(self=self, **kwargs)
        custom_required_parameters = ["timesteps", "embedding_dict", "sep_tokens", "make_negative_sample"]
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
            model_hyperparams = {k: v for k, v in model_hyperparams.items() if k in Bert.__init__.__code__.co_varnames}
            model = Bert(**model_hyperparams)
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

