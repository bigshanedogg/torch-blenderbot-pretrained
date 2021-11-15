import os
import re
import json
import shutil
import pickle
import torch
from collections import OrderedDict
from transformer.utils.common import is_primitive, init_path
from transformer.models.interface import ModelInterface

class ModelFilenameConstants:
    HYPERPARAMS_FILENAME_POSTFIX = "_hyperparams.pt"
    STATE_DICT_FILENAME_POSTFIX = "_state_dict.pt"
    MODEL_HYPERPARAMS_FILENAME = "model_hyperparams.pt"
    MODEL_STATE_DICT_FILENAME = "model_state_dict.pt"
    OPTIMIZER_HYPERPARAMS_FILENAME = "optimizer_hyperparams.pt"
    OPTIMIZER_STATE_DICT_FILENAME = "optimizer_state_dict.pt"
    HISTORY_FILENAME = "history.pickle"
    TRAIN_CONFIG_FILENAME = "train_config.json"
    EPOCH_LOG_FILENAME = "epoch_log.txt"
    BATCH_LOG_FILENAME = "batch_log.txt"
    TORCH_DDP_STATE_DICT_PREFIX = "module."
    # spm_model directory
    TOKENIZER_DIR = "tokenizer/"
    SPM_MODEL_DIR = "spm_model/"
    SRC_SPM_MODEL_DIR = "src_spm_model/"
    TGT_SPM_MODEL_DIR = "tgt_spm_model/"
    LEFT_SPM_MODEL_DIR = "left_spm_model/"
    RIGHT_SPM_MODEL_DIR = "right_spm_model/"
    # encoded data directory & filename
    ENCODED_DATA_DIRECTORY = "encoded_data/"
    DIALOG_HISTORY_SET_FILENAME = "dialog_history_set.pickle"
    DIALOG_RESPONSE_SET_FILENAME = "dialog_response_set.pickle"

def save_model(path, model, save_hyperparams=True, ddp=False):
    if not path.endswith("/"): path += "/"
    # save hyperparameters for initializing model
    if save_hyperparams:
        _save_model_hyperparams(path=path, model=model)
    # save model state_dict
    _save_model_state_dict(path=path, model=model, ddp=ddp)
    return path

def _save_model_hyperparams(path, model):
    path = init_path(path, reset=False)
    hyperparams_path = path + ModelFilenameConstants.MODEL_HYPERPARAMS_FILENAME
    checkpoint = dict()
    checkpoint["hyperparams"] = _extract_model_hyperparams(model=model)
    torch.save(checkpoint, hyperparams_path)

    for k,v in model._modules.items():
        if isinstance(v, ModelInterface):
            if k.startswith("module/"): k = re.sub("module/", "", k)
            if k.strip() == "": continue
            _path = path + k + "/"
            _save_model_hyperparams(path=_path, model=v)

def _extract_model_hyperparams(model):
    hyperparams = {k: v for k, v in model.__dict__.items() if (k in model.__init__.__code__.co_varnames) and is_primitive(obj=v)}
    return hyperparams

def _save_model_state_dict(path, model, ddp):
    path = init_path(path, reset=False)
    state_dict_path = path + ModelFilenameConstants.MODEL_STATE_DICT_FILENAME
    checkpoint = dict()
    checkpoint["state_dict"] = _extract_model_state_dict(model=model, ddp=ddp)
    torch.save(checkpoint, state_dict_path)

    for k,v in model._modules.items():
        if isinstance(v, ModelInterface):
            if k.startswith("module/"): k = re.sub("module/", "", k)
            if k.strip() == "": continue
            _path = path + k + "/"
            _save_model_state_dict(path=_path, model=v, ddp=ddp)

def _extract_model_state_dict(model, ddp):
    state_dict = {k: torch.clone(v).to("cpu") for k, v in model.state_dict().items()}
    if ddp:
        state_dict = remove_state_dict_prefix(state_dict=state_dict, prefix=ModelFilenameConstants.TORCH_DDP_STATE_DICT_PREFIX)
    return state_dict

def save_optimizer(path, optimizer, save_hyperparams=True):
    if not path.endswith("/"): path += "/"
    hyperparams_path = path + ModelFilenameConstants.OPTIMIZER_HYPERPARAMS_FILENAME
    state_dict_path = path + ModelFilenameConstants.OPTIMIZER_STATE_DICT_FILENAME

    # create checkpoint
    if save_hyperparams:
        checkpoint = dict()
        checkpoint["hyperparams"] = _extract_optimizer_hyperparams(optimizer=optimizer)
        torch.save(checkpoint, hyperparams_path)

    checkpoint = dict()
    checkpoint["state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, state_dict_path)
    return path

def _extract_optimizer_hyperparams(optimizer):
    hyperparams = {k: v for k, v in optimizer.__dict__.items() if is_primitive(obj=v)}
    return hyperparams

def save_history(path, history):
    if not path.endswith(ModelFilenameConstants.HISTORY_FILENAME):
        if not path.endswith("/"): path += "/"
        path = path + ModelFilenameConstants.HISTORY_FILENAME
    with open(path, "wb") as fp:
        pickle.dump(history, fp)
    return path

def save_config(path, config):
    if not path.endswith(ModelFilenameConstants.TRAIN_CONFIG_FILENAME):
        if not path.endswith("/"): path += "/"
        path = path + ModelFilenameConstants.TRAIN_CONFIG_FILENAME
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(config, fp)
    return path

def append_log(path, train_log_str, val_log_str, mode):
    if mode == "epoch":
        if not path.endswith(ModelFilenameConstants.EPOCH_LOG_FILENAME):
            if not path.endswith("/"): path += "/"
            path = path + ModelFilenameConstants.EPOCH_LOG_FILENAME
    elif mode == "batch":
        if not path.endswith(ModelFilenameConstants.BATCH_LOG_FILENAME):
            if not path.endswith("/"): path += "/"
            path = path + ModelFilenameConstants.BATCH_LOG_FILENAME
    with open(path, "a", encoding="utf-8") as fp:
        if train_log_str is not None:
            fp.write(train_log_str + "\n")
        if val_log_str is not None:
            fp.write(val_log_str + "\n")
    return path

def save_dialog_history_set(path, data):
    if not path.endswith("/"): path += "/"
    path = path + ModelFilenameConstants.ENCODED_DATA_DIRECTORY
    if not os.path.exists(path) or not os.path.isdir(path):  os.makedirs(path)
    path = path + ModelFilenameConstants.DIALOG_HISTORY_SET_FILENAME
    with open(path, "wb") as fp:
        pickle.dump(data, fp)
    return path

def save_dialog_response_set(path, data):
    if not path.endswith("/"): path += "/"
    path = path + ModelFilenameConstants.ENCODED_DATA_DIRECTORY
    if not os.path.exists(path) or not os.path.isdir(path):  os.makedirs(path)
    path = path + ModelFilenameConstants.DIALOG_RESPONSE_SET_FILENAME
    with open(path, "wb") as fp:
        pickle.dump(data, fp)
    return path

def load_model_hyperparams(path):
    if not path.endswith(ModelFilenameConstants.HYPERPARAMS_FILENAME_POSTFIX):
        path = path + ModelFilenameConstants.MODEL_HYPERPARAMS_FILENAME
    checkpoint = torch.load(path)
    hyperparams = checkpoint["hyperparams"]
    return hyperparams

def load_optimizer_hyperparams(path):
    if not path.endswith(ModelFilenameConstants.HYPERPARAMS_FILENAME_POSTFIX):
        path = path + ModelFilenameConstants.OPTIMIZER_HYPERPARAMS_FILENAME
    checkpoint = torch.load(path)
    hyperparams = checkpoint["hyperparams"]
    return hyperparams

def load_state_dict(object, path, map_location=None):
    if not path.endswith(ModelFilenameConstants.STATE_DICT_FILENAME_POSTFIX):
        if isinstance(object, torch.nn.modules.Module):
            path = path + ModelFilenameConstants.MODEL_STATE_DICT_FILENAME
        elif isinstance(object, torch.optim.Optimizer):
            path = path + ModelFilenameConstants.OPTIMIZER_STATE_DICT_FILENAME

    if map_location is None: map_location = torch.device("cpu")
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint["state_dict"]
    object.load_state_dict(state_dict)
    return object

def load_history(path):
    if not path.endswith(ModelFilenameConstants.HISTORY_FILENAME):
        path = path + ModelFilenameConstants.HISTORY_FILENAME
    history = None
    with open(path, "rb") as fp:
        history = pickle.load(fp)
    return history

def copy_epoch_log(from_path, to_path):
    if not from_path.endswith(ModelFilenameConstants.EPOCH_LOG_FILENAME):
        from_path = from_path + ModelFilenameConstants.EPOCH_LOG_FILENAME
    if not to_path.endswith(ModelFilenameConstants.EPOCH_LOG_FILENAME):
        to_path = to_path + ModelFilenameConstants.EPOCH_LOG_FILENAME
    if from_path != to_path:
        shutil.copyfile(src=from_path, dst=to_path)
    return to_path

def copy_batch_log(from_path, to_path):
    if not from_path.endswith(ModelFilenameConstants.BATCH_LOG_FILENAME):
        from_path = from_path + ModelFilenameConstants.BATCH_LOG_FILENAME
    if not to_path.endswith(ModelFilenameConstants.BATCH_LOG_FILENAME):
        to_path = to_path + ModelFilenameConstants.BATCH_LOG_FILENAME
    if from_path != to_path:
        shutil.copyfile(src=from_path, dst=to_path)
    return to_path

def load_encoded_context_set(path):
    if not path.endswith(ModelFilenameConstants.DIALOG_HISTORY_SET_FILENAME):
        if not path.endswith(ModelFilenameConstants.ENCODED_DATA_DIRECTORY):
            path = path + ModelFilenameConstants.ENCODED_DATA_DIRECTORY + ModelFilenameConstants.DIALOG_HISTORY_SET_FILENAME
        else:
            path = path + ModelFilenameConstants.DIALOG_HISTORY_SET_FILENAME
    data = None
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data

def load_encoded_candidate_set(path):
    if not path.endswith(ModelFilenameConstants.DIALOG_RESPONSE_SET_FILENAME):
        if not path.endswith(ModelFilenameConstants.ENCODED_DATA_DIRECTORY):
            path = path + ModelFilenameConstants.ENCODED_DATA_DIRECTORY + ModelFilenameConstants.DIALOG_RESPONSE_SET_FILENAME
        else:
            path = path + ModelFilenameConstants.DIALOG_RESPONSE_SET_FILENAME
    data = None
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data

def is_model_saved(path, save_hyperparms=True):
    if path is None: return False
    if not os.path.exists(path): return False
    if not os.path.isdir(path): return False
    if not os.path.exists(path + ModelFilenameConstants.MODEL_STATE_DICT_FILENAME): return False
    if save_hyperparms and not os.path.exists(path + ModelFilenameConstants.MODEL_HYPERPARAMS_FILENAME): return False
    return True

def is_optimizer_saved(path, save_hyperparms=True):
    if path is None: return False
    if not os.path.exists(path): return False
    if not os.path.isdir(path): return False
    if not os.path.exists(path + ModelFilenameConstants.OPTIMIZER_STATE_DICT_FILENAME): return False
    if save_hyperparms and not os.path.exists(path + ModelFilenameConstants.OPTIMIZER_HYPERPARAMS_FILENAME): return False
    return True

def is_history_saved(path):
    if path is None: return False
    if not os.path.exists(path): return False
    if not os.path.isdir(path): return False
    if not os.path.exists(path + ModelFilenameConstants.HISTORY_FILENAME): return False
    return True

def is_epoch_log_saved(path):
    if path is None: return False
    if not os.path.exists(path): return False
    if not os.path.isdir(path): return False
    if not os.path.exists(path + ModelFilenameConstants.EPOCH_LOG_FILENAME): return False
    return True

def is_batch_log_saved(path):
    if path is None: return False
    if not os.path.exists(path): return False
    if not os.path.isdir(path): return False
    if not os.path.exists(path + ModelFilenameConstants.BATCH_LOG_FILENAME): return False
    return True

def remove_state_dict_prefix(state_dict, prefix):
    output = OrderedDict()
    prefix = "^" + prefix
    for k,v in state_dict.items():
        k = re.sub(prefix, "", k)
        output[k] = v
    return output