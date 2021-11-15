import re
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from transformer.assertions.object_assertion import ServiceAssertion
from transformer.utils.common import get_device_index, get_available_devices, set_seed

class ServiceInterface(ServiceAssertion):
    device = "cpu"
    trainer = None
    model = None
    model_history = None
    preprocessor = None
    # constants
    temp_dir = None
    available_devices = []
    available_device_indice = []

    def __init__(self, temp_dir="./", verbose: bool = True, seed=20210830):
        set_seed(seed)
        # constant
        self.temp_dir = temp_dir
        self.verbose = verbose
        self.available_devices = get_available_devices()
        self.available_device_indice = [get_device_index(device) for device in self.available_devices]
        # instance setting
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_device(self, device:str):
        device_index = get_device_index(device)
        if device_index in self.available_device_indice:
            self.device = "cuda:{device_index}".format(device_index=device_index)
        else:
            self.device = "cpu"
        return self.device

    def load_model(self, model_dir, language="kor"):
        self.assert_implemented(method_name="load_model")

    def encode(self):
        self.assert_implemented(method_name="_encode")

    def sort_and_rank(self, candidates: List[str], probs: List[float], top_n: int, min_length: int, prev_utterance: str = None, intersection_tolerance: float = 0.5):
        prev_utterance_tokens = None
        if prev_utterance is not None:
            prev_utterance_tokens = set(self.tokenizer.encode(prev_utterance))

        _candidate_indice = np.argsort(probs)
        candidate_indice = _candidate_indice[::-1]

        sorted_candidates = candidates[candidate_indice]
        sorted_probs = probs[candidate_indice]

        output = []
        for candidate, prob in zip(sorted_candidates, sorted_probs):
            if len(self.tokenizer.tokenize(candidate)) < min_length: continue
            if prev_utterance_tokens is not None:
                intersection_len = len(set(candidate).intersection(prev_utterance_tokens))
                if intersection_len / len(set(candidate)) > intersection_tolerance: continue

            candidate = self.postprocess(text=candidate)
            row = (candidate, float(prob))
            output.append(row)
            if len(output) >= top_n: break
        return output


    def decode(self, candidates: List[List[int]], probs: List[float], min_length: int, top_n: int, prev_utterance: str = None, intersection_tolerance: float = 0.5):
        prev_utterance_tokens = None
        if prev_utterance is not None:
            prev_utterance_tokens = set(self.tokenizer.encode(prev_utterance))

        output = []
        for candidate, prob in zip(candidates, probs):
            if len(candidate) < min_length: continue
            if prev_utterance_tokens is not None:
                intersection_len = len(set(candidate).intersection(prev_utterance_tokens))
                if intersection_len / len(set(candidate)) > intersection_tolerance: continue

            candidate_output = self.tokenizer.decode(candidate, skip_special_tokens=True)
            candidate_output = self.postprocess(text=candidate_output)
            row = (candidate_output, float(prob))
            output.append(row)
            if len(output) >= top_n: break
        return output

    def postprocess(self, text):
        text = re.sub("\s+", " ", text)
        text = text.strip()
        return text

    def remove_dumplicate_candidates(self, candidates, candidate_embeds):
        new_candidates = []
        new_candidate_embeds = []
        for candidate, candidate_embed in zip(candidates, candidate_embeds):
            if candidate in new_candidates: continue
            new_candidates.append(candidate)
            new_candidate_embeds.append(candidate_embed)
        return new_candidates, new_candidate_embeds