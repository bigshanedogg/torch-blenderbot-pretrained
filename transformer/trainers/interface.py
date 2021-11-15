import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
import logging
import torch

from transformer.preprocessors.interface import PreprocessorInterface
from transformer.data.dataset import DatasetFromObject, DatasetFromDir, DatasetFromFile
from transformer.assertions.interface import Assertion
from transformer.assertions.object_assertion import TrainerAssertion
from transformer.utils.common import get_now_str, is_empty_row_in_dict, init_path, get_device_index
from transformer.trainers.utils import *

class TrainerInterface(TrainerAssertion):
    '''
    must implement following method 'create_model', 'iteration', 'get_ddp_params', 'get_criterions', 'get_criterion_weights', 'get_optimizer', 'get_data_loader', 'load'
    '''
    ngpus_per_node = None
    gpu_devices = None
    num_workers = None
    pin_memory = None
    epoch = 0
    step = 0
    lr_update = False
    lr_lambda = None
    # variables for specific trainer
    unlikelyhood_loss_update = False

    def __init__(self, temp_dir="./"):
        self.ngpus_per_node = 1 if torch.cuda.is_available() else 0
        self.num_workers = 2
        self.pin_memory = bool(self.ngpus_per_node)
        self.verbose_template = "{mode} ({device}) [{idx:^3d}/{num_iters:^3d}]:"
        if temp_dir.endswith("/"): temp_dir = temp_dir[:-1]
        self.temp_dir = "{model_path}/{now}/".format(model_path=temp_dir, now=get_now_str(str_format="%Y%m%d_%H%M%S"))
        self.mode_dir = "{mode}_{idx}/"
        print("'temp_dir' has been set to '{temp_dir}' to save model while training".format(temp_dir=self.temp_dir))
        self.keep_last = True

    def create_model(self, **params):
        self.assert_implemented(method_name="create_model")

    def create_data_loader(self, **params):
        self.assert_implemented(method_name="create_data_loader")

    def main_worker(self, device, config, preprocessor, ddp_params, train_params, data_params,
                    model_params, optimizer_params, criterion_params, data_loader_params):
        if self.gpu_devices is not None: device = self.gpu_devices[device]
        local_batch_size = int(train_params["batch_size"])
        local_num_workers = self.num_workers
        local_pin_memory = self.pin_memory

        # init c10d process_group
        print("From GPU Device {} ==> Initialize c10d process group..".format(device))
        init_method = "{protocol}://{host}:{port}".format(protocol=ddp_params["protocol"], host=ddp_params["host"], port=ddp_params["port"])
        torch.distributed.init_process_group(backend=ddp_params["backend"], init_method=init_method, world_size=self.ngpus_per_node, rank=device)

        print("From GPU Device {} ==> Making model..".format(device))
        # define model, criterions, and optimizer
        model = self.create_model(**model_params)
        criterions, criterion_weights = self.get_criterions(**model_params, **criterion_params)
        optimizer = self.get_optimizer(model=model, **optimizer_params)

        # load model.state_dict() when resume_path is not None
        if is_model_saved(path=train_params["resume_path"], save_hyperparms=False) and is_optimizer_saved(path=train_params["resume_path"], save_hyperparms=False):
            print("From GPU Device {} ==> Loading previous model from {}..".format(device, train_params["resume_path"]))
            map_location = "cuda:{}".format(device)
            model = load_state_dict(object=model, path=train_params["resume_path"], map_location=map_location)
            optimizer = load_state_dict(object=optimizer, path=train_params["resume_path"], map_location=map_location)
            if is_epoch_log_saved(path=train_params["resume_path"]):
                copy_epoch_log(from_path=train_params["resume_path"], to_path=self.temp_dir)
            if is_batch_log_saved(path=train_params["resume_path"]):
                copy_batch_log(from_path=train_params["resume_path"], to_path=self.temp_dir)

        if not self.lr_update:
            optimizer = self.update_optimizer_lr(optimizer=optimizer, lr=optimizer_params["initial_learning_rate"])
            print("From GPU Device {} ==> LearningRate has been set to {}..".format(device, optimizer_params["initial_learning_rate"]))
        model.train()

        # set device
        print("From GPU Device {} ==> Set gpu_device & DDP..".format(device))
        model = TrainerInterface.set_device(obj=model, device=device)
        optimizer = TrainerInterface.set_device(obj=optimizer, device=device)
        criterions = TrainerInterface.set_device(obj=criterions, device=device)
        # set DDP
        if model.is_composed_model():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

        # define data_loader
        print("From GPU Device {} ==> Preparing data_loader..".format(device))
        train_dataset = DatasetFromDir(data_dir=data_params["train_data_dir"], batch_size=local_batch_size, device=device, nprocs=ddp_params["nprocs"], encoding=data_params["encoding"], extension=data_params["extension"])
        train_data_loader_params = self.get_data_loader_params(dataset=train_dataset, preprocessor=preprocessor, batch_size=local_batch_size, device=device, nprocs=ddp_params["nprocs"], **data_loader_params)
        train_data_loader = self.create_data_loader(**train_data_loader_params)

        val_data_loader = None
        if data_params["val_data_dir"] is not None:
            val_dataset = DatasetFromDir(data_dir=data_params["val_data_dir"], batch_size=local_batch_size, device=device, nprocs=ddp_params["nprocs"], encoding=data_params["encoding"], extension=data_params["extension"])
            val_data_loader_params = self.get_data_loader_params(dataset=val_dataset, preprocessor=preprocessor, batch_size=local_batch_size, device=device, nprocs=ddp_params["nprocs"], **data_loader_params)
            val_data_loader = self.create_data_loader(**val_data_loader_params)

        # fit
        print("From GPU Device {} ==> Start training..".format(device))
        save_per_epoch = -1
        save_per_batch = -1
        verbose_per_epoch = -1
        verbose_per_batch = -1
        if device % self.ngpus_per_node == 0:
            save_per_epoch = train_params["save_per_epoch"]
            save_per_batch = train_params["save_per_batch"]
            verbose_per_epoch = train_params["verbose_per_epoch"]
            verbose_per_batch = train_params["verbose_per_batch"]
        else:
            val_data_loader = None

        history = self.fit(model=model, train_data_loader=train_data_loader, val_data_loader=val_data_loader,
                           criterions=criterions, criterion_weights=criterion_weights, optimizer=optimizer,
                           device=device, epoch=train_params["epoch"], amp=train_params["amp"],
                           save_per_epoch=save_per_epoch, save_per_batch=save_per_batch, keep_last=train_params["keep_last"],
                           verbose_per_epoch=verbose_per_epoch, verbose_per_batch=verbose_per_batch)

        # save model.state_dict()
        if device % self.ngpus_per_node == 0:
            save_path = train_params["save_path"]
            if not save_path.endswith("/"): save_path = save_path + "/"
            print("From GPU Device {} ==> Saving model.state_dict() into {}..".format(device, save_path))
            TrainerInterface.save(path=save_path, model=model, optimizer=optimizer, history=history, config=config, preprocessor=preprocessor,
                                  save_model_hyperparams=False, save_optimizer_hyperparams=False, ddp=True)
            save_additional_params = {
                "model": model,
                "train_data_loader": train_data_loader,
                "val_data_loader": val_data_loader,
                "device": device
            }
            self.save_additional(path=save_path, **save_additional_params)
        return history

    def save_additional(self, path, **kwargs):
        return path

    def fit(self, model, train_data_loader: torch.utils.data.DataLoader, val_data_loader: torch.utils.data.DataLoader, criterions, criterion_weights, optimizer, device, epoch: int = 1, amp: bool = False,
            save_per_epoch: int = 1, save_per_batch: int = -1, keep_last: bool = True, verbose_per_epoch: int = -1, verbose_per_batch: int = -1):
        # fit
        train_history = TrainHistory()
        val_history = TrainHistory()
        self.keep_last = keep_last
        if self.ngpus_per_node == 1 or get_device_index(device) == 0:
            init_path(self.temp_dir, reset=False)
        epoch_train_history_str = None
        epoch_val_history_str = None

        # define scaler
        scaler = None
        if amp: scaler = torch.cuda.amp.GradScaler()
        for epoch_idx in range(1, epoch + 1):
            # if self.unlikelyhood_loss_update: self.init_prev_token_distribution()
            train_verbose = self.verbose_template[:-1].format(mode="Epoch_train", device=device, idx=epoch_idx, num_iters=epoch)
            # train iteration
            train_data_iter = tqdm(train_data_loader, initial=train_data_loader.iter_start, total=len(train_data_loader), desc=train_verbose, bar_format="{l_bar}{r_bar}")
            train_data_iter.iter_size = train_data_loader.iter_end - train_data_loader.iter_start
            epoch_train_history = self.train_epoch(model=model, data_loader=train_data_iter, criterions=criterions, criterion_weights=criterion_weights, optimizer=optimizer, device=device, amp=amp, scaler=scaler, save_per_batch=save_per_batch, verbose_per_batch=verbose_per_batch)
            epoch_train_history_str = self.verbose_template.format(mode="\nEpoch_train", device=device, idx=epoch_idx, num_iters=epoch) + str(epoch_train_history)
            # update train_history
            train_history += epoch_train_history

            # verbose
            if verbose_per_epoch > 0 and epoch_idx % verbose_per_epoch == 0:
                print(epoch_train_history_str)

            if val_data_loader is not None:
                val_verbose = self.verbose_template[:-1].format(mode="Epoch_val", device=device, idx=epoch_idx, num_iters=epoch)
                # val iteration
                val_data_iter = tqdm(val_data_loader, initial=val_data_loader.iter_start, total=len(val_data_loader), desc=val_verbose, bar_format="{l_bar}{r_bar}")
                val_data_iter.iter_size = val_data_loader.iter_end - val_data_loader.iter_start
                epoch_val_history = self.test_epoch(model=model, data_loader=val_data_iter, criterions=criterions, criterion_weights=criterion_weights, optimizer=optimizer, device=device, amp=amp, scaler=scaler, verbose_per_batch=-1)
                epoch_val_history_str = self.verbose_template.format(mode="\nEpoch_val", device=device, idx=epoch_idx, num_iters=epoch) + str(epoch_val_history)
                # update val_history
                val_history += epoch_val_history
                # verbose
                if verbose_per_epoch > 0 and epoch_idx % verbose_per_epoch == 0:
                    print(epoch_val_history_str)


            if save_per_epoch > 0 and epoch_idx % save_per_epoch == 0:
                # remove previous model if keep_last is True
                prev_epoch_idx = epoch_idx - save_per_epoch
                prev_temp_path = self.temp_dir + self.mode_dir.format(mode="epoch", idx=prev_epoch_idx)
                if keep_last and os.path.exists(prev_temp_path): shutil.rmtree(prev_temp_path)
                # save
                history = dict()
                history["train"] = train_history.to_dict()
                history["val"] = val_history.to_dict()
                temp_path = self.temp_dir + self.mode_dir.format(mode="epoch", idx=epoch_idx)
                TrainerInterface.save(path=temp_path, model=model, optimizer=optimizer, history=history, config=None,  preprocessor=train_data_loader.preprocessor,
                                      save_model_hyperparams=False, save_optimizer_hyperparams=False, ddp=True)

            # write log to temp_dir
            if self.ngpus_per_node == 1 or get_device_index(device) == 0:
                append_log(path=self.temp_dir, train_log_str=epoch_train_history_str, val_log_str=epoch_val_history_str, mode="epoch")
                epoch_train_history_str = None
                epoch_val_history_str = None

        history = dict()
        history["train"] = train_history.to_dict()
        history["val"] = val_history.to_dict()
        return history

    def train_epoch(self, model, data_loader: torch.utils.data.DataLoader, criterions, criterion_weights, optimizer, device,
                    amp: bool = False, scaler: torch.cuda.amp.GradScaler = None, save_per_batch: int = -1, verbose_per_batch: int = -1):
        epoch_history = TrainHistory()
        batch_history = TrainHistory()
        batch_history_str = None

        model.train()
        with torch.set_grad_enabled(True):
            for batch_idx, batch in enumerate(data_loader):
                batch_idx += 1
                batch = [{k: self.convert_to_tensor(data=v, device=device) for k, v in _batch.items()} for _batch in batch]

                is_empty_flag = [is_empty_row_in_dict(data=_batch) for _batch in batch]
                if any(is_empty_flag): continue
                # if is_empty_row_in_dict(data=inputs) or is_empty_row_in_dict(data=outputs): continue
                # inputs = {k: self.convert_to_tensor(data=v, device=device) for k, v in inputs.items()}
                # outputs = {k: self.convert_to_tensor(data=v, device=device) for k, v in outputs.items()}
                # iteration
                loss_dict, acc_dict = self.iteration(model=model, batch=batch,
                                                     criterions=criterions, criterion_weights=criterion_weights,
                                                     optimizer=optimizer, train=True, amp=amp, scaler=scaler)
                # update train_result instance
                batch_history.update(loss_dict=loss_dict, acc_dict=acc_dict, lr=optimizer.param_groups[0]["lr"])
                # verbose
                if verbose_per_batch > 0 and batch_idx % verbose_per_batch == 0:
                    batch_history_str = self.verbose_template.format(mode="\nBatch_train", device=device, idx=batch_idx, num_iters=len(data_loader)) + str(batch_history)
                    print(batch_history_str)

                    # write log to temp_dir
                    if self.ngpus_per_node == 1 or get_device_index(device) == 0:
                        append_log(path=self.temp_dir, train_log_str=batch_history_str, val_log_str=None, mode="batch")
                        batch_history_str = None

                    epoch_history += batch_history
                    batch_history = TrainHistory()

                # save per batch
                if save_per_batch > 0 and batch_idx % save_per_batch == 0:
                    prev_batch_idx = batch_idx - save_per_batch
                    prev_temp_path = self.temp_dir + self.mode_dir.format(mode="batch", idx=prev_batch_idx)
                    if self.keep_last and os.path.exists(prev_temp_path): shutil.rmtree(prev_temp_path)
                    temp_path = self.temp_dir + self.mode_dir.format(mode="batch", idx=batch_idx)
                    temp_path = init_path(path=temp_path, reset=True)
                    TrainerInterface.save(path=temp_path, model=model, optimizer=optimizer, history=epoch_history.to_dict(), config=None,  preprocessor=data_loader.preprocessor,
                                          save_model_hyperparams=True, save_optimizer_hyperparams=False, ddp=True)
                if batch_idx > data_loader.iter_size: break

            if batch_history.iteration > 0: epoch_history += batch_history
        self.epoch += 1
        return epoch_history

    def test_epoch(self, model, data_loader: torch.utils.data.DataLoader, criterions, criterion_weights, optimizer, device, amp: bool = False, scaler: torch.cuda.amp.GradScaler = None, verbose_per_batch: int = -1):
        epoch_history = TrainHistory()
        batch_history = TrainHistory()

        model.eval()
        with torch.set_grad_enabled(False):
            for batch_idx, batch in enumerate(data_loader):
                batch_idx += 1
                batch = [{k: self.convert_to_tensor(data=v, device=device) for k, v in _batch.items()} for _batch in batch]

                is_empty_flag = [is_empty_row_in_dict(data=_batch) for _batch in batch]
                if any(is_empty_flag): continue

                # iteration
                loss_dict, acc_dict = self.iteration(model=model, batch=batch,
                                                     criterions=criterions, criterion_weights=criterion_weights,
                                                     optimizer=optimizer, train=False, amp=amp, scaler=scaler)
                # update train_result instance
                batch_history.update(loss_dict=loss_dict, acc_dict=acc_dict, lr=optimizer.param_groups[0]["lr"])
                # verbose
                if verbose_per_batch > 0 and batch_idx % verbose_per_batch == 0:
                    batch_history_str = self.verbose_template.format(mode="\nBatch_val", device=device, idx=batch_idx, num_iters=len(data_loader)) + str(batch_history)
                    print(batch_history_str)

                    # write log to temp_dir
                    if self.ngpus_per_node == 1 or get_device_index(device) == 0:
                        append_log(path=self.temp_dir, train_log_str=None, val_log_str=batch_history_str, mode="batch")
                        batch_history_str = None

                    epoch_history += batch_history
                    batch_history = TrainHistory()
                if batch_idx > data_loader.iter_size: break

            if batch_history.iteration > 0: epoch_history += batch_history
        return epoch_history

    def get_init_params(self, required_parameters, default_parameters, kwargs):
        self.assert_contain_elements(required=required_parameters, target=kwargs, name="required_parameters")

        params = dict()
        for k, v in default_parameters:
            if k in kwargs: v = kwargs.pop(k)
            params[k] = v
        for k, v in kwargs.items():
            if k not in required_parameters: continue
            params[k] = v
        return params

    def iteration(self, model, inputs, outputs, criterions, criterion_weights, optimizer, train):
        self.assert_implemented(method_name="iteraion")

    def inference(self):
        self.assert_implemented(method_name="inference")

    def update_optimizer_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return optimizer

    def convert_to_tensor(self, data, device):
        if isinstance(data, list):
            data = np.array(data)
        data = torch.from_numpy(data).to(device)
        return data

    def convert_to_numpy(self, tensor: torch.Tensor):
        tensor = tensor.cpu().detach().numpy()
        return tensor

    @classmethod
    def set_device(cls, obj, device):
        if isinstance(obj, torch.nn.modules.Module):
            obj = cls._set_model_device(model=obj, device=device)
        elif isinstance(obj, torch.optim.Optimizer):
            obj = cls._set_optimizer_device(optimizer=obj, device=device)
        elif isinstance(obj, dict):
            obj = cls._set_criterions_device(criterions=obj, device=device)
        return obj

    @classmethod
    def _set_model_device(cls, model, device):
        if device is None: device = "cpu"
        if device == "cpu" or (isinstance(device, torch.device) and device.type == "cpu"):
            non_blocking = False
        else:
            torch.cuda.set_device(device)
            non_blocking = True

        # set device
        model.to(device, non_blocking=non_blocking)

        model.device = device
        message = "Setting model device: {device}".format(device=device)
        print(message)
        return model

    @classmethod
    def _set_optimizer_device(cls, optimizer, device):
        for param in optimizer.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)
        return optimizer

    @classmethod
    def _set_criterions_device(cls, criterions, device):
        if device is None:
            device = "cpu"

        # set device
        for k,v in criterions.items():
            v.set_device(device=device)
            criterions[k] = v

        message = "Setting criterions device: {device}".format(device=device)
        print(message)
        return criterions

    def set_ngpus_per_node(self, ngpus_per_node, gpu_devices=None):
        self.assert_equal_or_lesser(value=ngpus_per_node, criteria=torch.cuda.device_count())
        self.ngpus_per_node = ngpus_per_node
        self.pin_memory = bool(ngpus_per_node)
        if gpu_devices is not None:
            gpu_devices = [int(device) if isinstance(device, str) else device for device in gpu_devices]
            self.gpu_devices = gpu_devices

    def get_criterions(self):
        self.assert_implemented(method_name="get_criterions")

    def get_optimizer(self, optimizer: str, model: torch.nn.Module.modules, **kwargs):
        self.assert_isin_optimizers(optimizer=optimizer)
        if optimizer == "transformer":
            required_parameters = ["initial_learning_rate", "beta_1", "beta_2", "optimizer_epsilon"]
            self.assert_contain_elements_(required=required_parameters, target=kwargs, name="parameters")
            initial_learning_rate = kwargs["initial_learning_rate"]
            beta_1 = kwargs["beta_1"]
            beta_2 = kwargs["beta_2"]
            optimizer_epsilon = kwargs["optimizer_epsilon"]
            optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate, betas=(beta_1, beta_2), eps=optimizer_epsilon)
        elif optimizer == "adam_w":
            required_parameters = ["initial_learning_rate"]
            self.assert_contain_elements_(required=required_parameters, target=kwargs, name="parameters")
            initial_learning_rate = kwargs["initial_learning_rate"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=initial_learning_rate)

        if self.assert_isinstance_optimizer(optimizer=optimizer):
            optimizer.__dict__.update(kwargs)
        return optimizer

    @classmethod
    def save(cls, path, model, optimizer, history, config=None, preprocessor=None, save_model_hyperparams=False, save_optimizer_hyperparams=False, ddp=False):
        path = init_path(path, reset=True)

        if model is not None:
            save_model(path=path, model=model, save_hyperparams=save_model_hyperparams, ddp=ddp)
        if optimizer is not None:
            save_optimizer(path=path, optimizer=optimizer, save_hyperparams=save_optimizer_hyperparams)
        if history is not None:
            save_history(path=path, history=history)
        if config is not None:
            save_config(path=path, config=config)
        if preprocessor is not None and isinstance(preprocessor, PreprocessorInterface):
            pass
            # ModelFilenname constant 바꾸고, 구현하기 
            # preprocessor.save_tokenizer(path=path) # TODO
        message = "Saved into {path}".format(path=path)
        print(message)
        logging.info(message)
        return path

    def load(self, path):
        self.assert_implemented(method_name="load_model")

    # parameters to give main_worker while DDP training
    @classmethod
    def get_ddp_params(cls, **kwargs):
        # parameters for dist.init_process_group method
        required_parameters = ["protocol", "host", "port", "backend", "nprocs"]
        cls.assert_contain_elements_(required=required_parameters, target=kwargs, name="parameters")
        params = dict()
        for k, v in kwargs.items(): params[k] = v
        return params

    @classmethod
    def get_train_params(cls, **kwargs):
        # parameters for trainer.fit method
        required_parameters = ["epoch", "batch_size", "amp", "save_path", "resume_path",
                               "save_per_epoch", "save_per_batch", "keep_last", "verbose_per_epoch", "verbose_per_batch"]
        cls.assert_contain_elements_(required=required_parameters, target=kwargs, name="parameters")
        params = dict()
        for k,v in kwargs.items(): params[k] = v
        return params

    @classmethod
    def get_data_params(cls, **kwargs):
        required_parameters = ["train_data_dir", "val_data_dir", "uncased", "encoding", "extension"]
        cls.assert_contain_elements_(required=required_parameters, target=kwargs, name="parameters")
        params = dict()
        for k, v in kwargs.items(): params[k] = v
        return params

    def get_data_loader_params(self, **params):
        # parameters for trainer.get_data_loader method
        common_required_parameters = ["dataset", "preprocessor", "batch_size", "device", "nprocs"]
        self.assert_contain_elements(required=common_required_parameters, target=params, name="common_required_parameters")
        return common_required_parameters


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