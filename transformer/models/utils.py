import re
import os
import json
import pickle
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, List, Any
import torch
from transformer.tokenizer.utils import load_tokenizer_from_pretrained
from transformer.utils.common import init_path

def get_length_penalty(length, alpha=1.2, min_length=5):
    # multiply output to cumulative_prob
    output = ((min_length + 1) / (min_length + length)) ** alpha
    return output

class ModelFilenameConstants:
    HYPERPARAMS_FILENAME_POSTFIX = "_hyperparams.pt"
    STATE_DICT_FILENAME_POSTFIX = "_state_dict.pt"
    MODEL_HYPERPARAMS_FILENAME = "model_hyperparams.pt"
    MODEL_STATE_DICT_FILENAME = "model_state_dict.pt"
    OPTIMIZER_HYPERPARAMS_FILENAME = "optimizer_hyperparams.pt"
    OPTIMIZER_STATE_DICT_FILENAME = "optimizer_state_dict.pt"
    HISTORY_FILENAME = "history.pickle"
    EPOCH_LOG_FILENAME = "epoch_log.txt"
    BATCH_LOG_FILENAME = "batch_log.txt"
    TORCH_DDP_STATE_DICT_PREFIX = "module."
    # spm_model directory
    TOKENIZER_DIR = "tokenizer/"
    # encoded data directory & filename
    ENCODED_DATA_DIRECTORY = "encoded_data/"
    CANDIDATES_FILENAME = "candidates.pickle"

def save_model(path, model, save_nested_model: bool = False, ddp: bool = False):
    if not path.endswith("/"): path += "/"
    path = init_path(path, reset=False)
    # save model state_dict
    _save_model_state_dict(path=path, model=model, save_nested_model=save_nested_model, ddp=ddp)
    return path

def _save_model_state_dict(path, model, save_nested_model, ddp):
    path = init_path(path, reset=False)
    state_dict_path = path + ModelFilenameConstants.MODEL_STATE_DICT_FILENAME
    checkpoint = dict()
    checkpoint["state_dict"] = _extract_model_state_dict(model=model, ddp=ddp)
    torch.save(checkpoint, state_dict_path)

    if save_nested_model:
        for k,v in model._modules.items():
            if isinstance(v, torch.nn.modules.Module):
                if k.startswith("module/"): k = re.sub("module/", "", k)
                if k.strip() == "": continue
                _path = path + k + "/"
                _save_model_state_dict(path=_path, model=v, ddp=ddp)

def _extract_model_state_dict(model, ddp):
    state_dict = {k: torch.clone(v).to("cpu") for k, v in model.state_dict().items()}
    if ddp:
        state_dict = remove_state_dict_prefix(state_dict=state_dict, prefix=ModelFilenameConstants.TORCH_DDP_STATE_DICT_PREFIX)
    return state_dict

def save_optimizer(path, optimizer):
    if not path.endswith("/"): path += "/"
    state_dict_path = path + ModelFilenameConstants.OPTIMIZER_STATE_DICT_FILENAME
    checkpoint = dict()
    checkpoint["state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, state_dict_path)
    return path

def save_tokenizer(path, tokenizer):
    if not path.endswith("/"): path += "/"
    tokenizer_path = path + ModelFilenameConstants.TOKENIZER_DIR
    tokenizer.save_pretrained(tokenizer_path)
    return path

def save_history(path, history):
    if not path.endswith(ModelFilenameConstants.HISTORY_FILENAME):
        if not path.endswith("/"): path += "/"
        path = path + ModelFilenameConstants.HISTORY_FILENAME
    with open(path, "wb") as fp:
        pickle.dump(history, fp)
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

def save_candidates(path, data):
    if not path.endswith("/"): path += "/"
    path = path + ModelFilenameConstants.ENCODED_DATA_DIRECTORY
    if not os.path.exists(path) or not os.path.isdir(path):  os.makedirs(path)
    path = path + ModelFilenameConstants.CANDIDATES_FILENAME
    with open(path, "wb") as fp:
        pickle.dump(data, fp)
    return path

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

def load_tokenizer(path, model_type: str):
    if not path.endswith("/"): path += "/"
    path = path + ModelFilenameConstants.TOKENIZER_DIR
    tokenizer = load_tokenizer_from_pretrained(model_type=model_type, name_or_path=path)
    return tokenizer

def load_candidates(path):
    if not path.endswith("/"): path += "/"
    path = path + ModelFilenameConstants.ENCODED_DATA_DIRECTORY + ModelFilenameConstants.CANDIDATES_FILENAME
    data = None
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data

def remove_state_dict_prefix(state_dict, prefix):
    output = OrderedDict()
    prefix = "^" + prefix
    for k,v in state_dict.items():
        k = re.sub(prefix, "", k)
        output[k] = v
    return output

def compute_bleu(metric, tokenizer, predictions: List[str], references: List[str]):
    predictions = [tokenizer.tokenize(prediction) for prediction in predictions]
    references = [[tokenizer.tokenize(reference)] for reference in references]
    score = metric.compute(predictions=predictions, references=references)
    return score

def compute_meteor(metric, tokenizer, predictions: List[str], references: List[str]):
    score = metric.compute(predictions=predictions, references=references)
    return score

def compute_rouge(metric, tokenizer, predictions: List[str], references: List[str]):
    predictions = [" ".join([str(_id) for _id in tokenizer.encode(prediction)]) for prediction in predictions]
    references = [" ".join([str(_id) for _id in tokenizer.encode(reference)]) for reference in references]
    score = metric.compute(predictions=predictions, references=references)
    return score

def compute_semantic_score(metric, tokenizer, predictions: List[str], references: List[str]):
    scores = metric.score(references=references, candidates=predictions, verbose=False)
    score = np.mean(scores)
    return score

def compute_hits(predictions: List[List[Any]], references: List[Any], k: List[int] = [1,2,5,10]):
    _score_list = [[] for i in range(0, len(k))]
    for prediction, reference in zip(predictions, references):
        for _k_idx, _k in enumerate(k):
            _score = 0
            if reference in prediction[:_k]:
                _score = 1
            _score_list[_k_idx].append(_score)

    score = []
    for _k_idx, _k in enumerate(k):
        score.append(np.mean(_score_list[_k_idx]))
    return score

def get_score_json(model_name, dataset_name, test_data_size, batch_size, scores):
    output = {
        "model": model_name,
        "dataset": dataset_name,
        "fine-tuning": False,
        "retriever_blending": False,
        "epoch": 0,
        "lr": -1,
        "test_data_size": test_data_size,
        "test_steps": (test_data_size // batch_size) + 1,
        "batch_size": batch_size,
        "metrics": {
        }
    }

    for k, v in scores.items():
        output["metrics"][k] = v

    return output