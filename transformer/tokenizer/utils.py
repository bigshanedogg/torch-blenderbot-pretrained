import os
import json
import nltk
import pickle
import shutil
import logging
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ElectraTokenizer


special_token_dict = None
with open("./config/special_token_dict.json", encoding="UTF-8") as fp:
    special_token_dict = json.load(fp)

def make_custom_tokenizer_from_pretrained(model_type:str, name_or_path: str, add_special_token: bool = True) -> PreTrainedTokenizerFast:
    tokenizer = None
    _special_token_dict = special_token_dict.copy()

    def get_tokenizer_params(base_special_tokens, _special_token_dict, add_special_token):
        special_token_params = dict()
        for base_special_token in base_special_tokens:
            k = base_special_token + "_token"
            v = _special_token_dict.pop(base_special_token)["token"]
            special_token_params[k] = v

        additional_special_token_params = dict()
        if add_special_token:
            for special_token_name, special_token in _special_token_dict.items():
                k = special_token_name + "_token"
                additional_special_token_params[k] = special_token["token"]
        return special_token_params, additional_special_token_params

    if model_type in ["gpt2", "bart"]:
        base_special_tokens = ["pad", "bos", "eos", "unk", "mask"]
        special_token_params, additional_special_token_params = get_tokenizer_params(base_special_tokens=base_special_tokens, _special_token_dict=_special_token_dict, add_special_token=add_special_token)
        additional_special_tokens = list(additional_special_token_params.values())
        tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=name_or_path,
                                                            additional_special_tokens=additional_special_tokens, **special_token_params)
    elif model_type in ["electra"]:
        base_special_tokens = ["pad", "cls", "sep", "unk", "mask"]
        pretrained_special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
        special_token_params, additional_special_token_params = get_tokenizer_params(base_special_tokens=base_special_tokens, _special_token_dict=_special_token_dict, add_special_token=add_special_token)
        additional_special_tokens = list(additional_special_token_params.values())
        for base_special_token, pretrained_special_token in zip(base_special_tokens, pretrained_special_tokens):
            k = base_special_token + "_token"
            special_token_params[k] = pretrained_special_token
        tokenizer = ElectraTokenizer.from_pretrained(pretrained_model_name_or_path=name_or_path,
                                                     additional_special_tokens=additional_special_tokens,
                                                     **special_token_params)

    # update unregistered special_tokens to class_variables
    tokenizer = update_special_token_vars(tokenizer=tokenizer)

    message = "loaded pretrained huggingface_tokenizer: '{name_or_path}'".format(name_or_path=name_or_path)
    print(message)
    return tokenizer


def load_tokenizer_from_pretrained(model_type:str, name_or_path: str) -> PreTrainedTokenizerFast:
    tokenizer = None
    if model_type in ["gpt2", "bart"]:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=name_or_path)
    elif model_type in ["electra"]:
        tokenizer = ElectraTokenizer.from_pretrained(pretrained_model_name_or_path=name_or_path)

    # update unregistered special_tokens to class_variables
    tokenizer = update_special_token_vars(tokenizer=tokenizer)

    message = "loaded pretrained huggingface_tokenizer: '{name_or_path}'".format(name_or_path=name_or_path)
    print(message)
    return tokenizer

def update_special_token_vars(tokenizer):
    dict_to_update = dict()
    _special_token_dict = special_token_dict.copy()
    if "additional_special_tokens" in tokenizer.special_tokens_map:
        for special_token, special_id in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids):
            if special_token not in tokenizer.special_tokens_map["additional_special_tokens"]: continue
            for k, v in _special_token_dict.items():
                if special_token == v["token"]:
                    dict_to_update[k+"_token"] = special_token
                    dict_to_update[k+"_token_id"] = special_id
        print("update unregistered special_tokens to class_variables:{}".format(dict_to_update))
        tokenizer.__dict__.update(dict_to_update)
    return tokenizer