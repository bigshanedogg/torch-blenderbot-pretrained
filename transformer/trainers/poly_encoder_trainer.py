import logging
from typing import Dict, List
from transformer.trainers.interface import TrainerInterface
from transformer.trainers.learning_rate_lambda import LearningRateLambda
from transformer.losses.criterions import LogLikelyhoodCriterion
from transformer.models.bert import Bert
from transformer.models._poly_encoder import PolyEncoder
from transformer.data.bert_data_loader import BertDataLoader
from transformer.trainers.utils import *
from transformer.utils.common import is_valid_file


class PolyEncoderTrainer(TrainerInterface):
    def __init__(self, temp_dir="./"):
        TrainerInterface.__init__(self, temp_dir=temp_dir)

    def get_init_params(self, **kwargs):
        required_parameters = ["context_encoder", "candidate_encoder", "m_code", "aggregation_method"]
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="required_parameters")
        default_parameters = []
        params = TrainerInterface.get_init_params(self=self, required_parameters=required_parameters, default_parameters=default_parameters, kwargs=kwargs)
        return params

    def create_encoder(self, model_type, encoder_model_path=None):
        # assert
        self.assert_isin_models(model=model_type)
        self.assert_is_valid_path(path=encoder_model_path)
        config_file_path = encoder_model_path + ModelFilenameConstants.TRAIN_CONFIG_FILENAME
        hyperparams_file_path = encoder_model_path + ModelFilenameConstants.MODEL_HYPERPARAMS_FILENAME
        assert_message = "Both {config_file_path} and {hyperparams_file_path} not exist.".format(config_file_path=config_file_path, hyperparams_file_path=hyperparams_file_path)

        is_config_file_exist = is_valid_file(path=config_file_path)
        is_hyperparams_file_exist = is_valid_file(path=hyperparams_file_path)
        assert is_config_file_exist or is_hyperparams_file_exist, assert_message

        encoder_hyperparams = None
        if is_hyperparams_file_exist:
            encoder_hyperparams = load_model_hyperparams(path=hyperparams_file_path)
        elif is_config_file_exist:
            with open(config_file_path, "r", encoding="utf-8") as fp:
                config = json.load(fp)
                encoder_hyperparams = dict()
                encoder_hyperparams["pad_token_id"] = 0
                encoder_hyperparams.update(config["model"])
                # encoder_hyperparams = config["model"]

        encoder = None
        if model_type == "bert":
            encoder = Bert(**encoder_hyperparams)

        return encoder

    def create_model(self, **kwargs):
        required_parameters = ["context_encoder", "candidate_encoder", "m_code", "aggregation_method"]
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="required_parameters")
        params = dict()
        for k,v in kwargs.items():
            if k not in required_parameters: continue
            params[k] = v

        # context_encoder
        context_encoder = params.pop("context_encoder")
        if isinstance(context_encoder, ModelInterface): pass
        elif isinstance(context_encoder, dict):
            context_encoder_model_type = context_encoder["model_type"]
            context_encoder_model_path = context_encoder["model_path"]
            context_encoder = self.create_encoder(model_type=context_encoder_model_type, encoder_model_path=context_encoder_model_path)

        # candidate_encoder
        candidate_encoder = params.pop("candidate_encoder")
        if isinstance(candidate_encoder, ModelInterface): pass
        elif isinstance(candidate_encoder, dict):
            candidate_encoder_model_type = candidate_encoder["model_type"]
            candidate_encoder_model_path = candidate_encoder["model_path"]
            candidate_encoder = self.create_encoder(model_type=candidate_encoder_model_type, encoder_model_path=candidate_encoder_model_path)

        # poly_encoder
        model = PolyEncoder(context_encoder=context_encoder, candidate_encoder=candidate_encoder, **params)
        return model

    def create_data_loader(self, **params):
        data_loader = BertDataLoader(**params)
        return data_loader

    def iteration(self, model, batch: List[Dict[str, torch.Tensor]], criterions, criterion_weights, optimizer, train: bool, amp: bool = False, scaler: torch.cuda.amp.GradScaler = None):
        loss = None
        context_inputs, candidate_inputs, outputs = batch
        with torch.cuda.amp.autocast(enabled=amp):
            # forward
            predictions = model.forward(context_inputs=context_inputs, candidate_inputs=candidate_inputs)

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
        if "ce" in kwargs: ce = kwargs.pop("ce")
        else: ce = 0.0
        if "reg" in kwargs: reg = kwargs.pop("reg")
        else: reg = 0.0
        if "cos" in kwargs: cos = kwargs.pop("cos")
        else: cos = 0.0
        if "l1" in kwargs: l1 = kwargs.pop("l1")
        else: l1 = 0.0
        if "l2" in kwargs: l2 = kwargs.pop("l2")
        else: l2 = 0.0

        self.assert_equal(a=1.0, b=(ce + reg + cos + l1 + l2))
        criterions = dict()
        criterion_weights = dict()
        if ce > 0.0:
            criterions["ce"] = LogLikelyhoodCriterion()
            criterion_weights["ce"] = ce
        if reg > 0.0:
            criterions["reg"] = LogLikelyhoodCriterion() # TODO
            criterion_weights["reg"] = reg
        if cos > 0.0:
            criterions["cos"] = LogLikelyhoodCriterion() # TODO
            criterion_weights["cos"] = cos
        if l1 > 0.0:
            criterions["l1"] = LogLikelyhoodCriterion() # TODO
            criterion_weights["sop"] = l1
        if l2 > 0.0:
            criterions["l2"] = LogLikelyhoodCriterion() # TODO
            criterion_weights["l2"] = l2
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
            model_hyperparams = load_optimizer_hyperparams(path=path + ModelFilenameConstants.MODEL_HYPERPARAMS_FILENAME)
            model = PolyEncoder(**model_hyperparams)
            model = load_state_dict(object=model, path=path + ModelFilenameConstants.MODEL_STATE_DICT_FILENAME)
            output["model"] = model

        # load optimizer
        if is_optimizer_saved(path=path, save_hyperparms=True):
            optimizer_hyperparams = load_model_hyperparams(path=path + ModelFilenameConstants.OPTIMIZER_HYPERPARAMS_FILENAME)
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