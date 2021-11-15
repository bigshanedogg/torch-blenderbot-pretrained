import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from transformer.data.generator_dataset import BlenderBotEncoder
from transformer.data.utils import merge_utterances
from transformer.models.gpt import Gpt2
from transformer.models.utils import ModelFilenameConstants, load_state_dict, load_tokenizer
from transformer.utils.common import convert_to_tensor
from transformer.services.interface import ServiceInterface

class Gpt2DialogGenerator(ServiceInterface):
    def load_model(self, model_dir):
        if not model_dir.endswith("/"): model_dir += "/"
        self.timesteps = 128
        self.encoder = BlenderBotEncoder()

        print("loading tokenizer...")
        self.tokenizer = load_tokenizer(path=model_dir, model_type="gpt2")
        self.vocab_size = len(self.tokenizer)
        if "context_token_id" in self.tokenizer.__dict__: self.context_token_id = self.tokenizer.context_token_id
        else: self.context_token_id = None
        if "condition_token_id" in self.tokenizer.__dict__: self.condition_token_id = self.tokenizer.condition_token_id
        else: self.condition_token_id = None
        print("loading model...")
        self.model = Gpt2(vocab_size=self.vocab_size)
        self.model = self.model.to(self.device)
        print("loading model_state_dict...")
        self.model = load_state_dict(object=self.model, path=model_dir)
        return model_dir

    def infer_next_utterance_greedy(self, utterances: List[str], speaker_ids: List[str], conditions: List[str],
                                    min_length: int, repetition_penalty: float, no_repeat_ngram_size: int,
                                    prev_utterance: str, intersection_tolerance: float, max_retry: int, return_probs: bool):
        input_ids, type_ids = self.encode(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions, max_retry=max_retry)
        begin_idx = len(input_ids[0])
        candidates = self.model.generate(input_ids,
                                        min_length=min_length,
                                        max_length=self.timesteps,
                                        repetition_penalty=repetition_penalty,
                                        no_repeat_ngram_size=no_repeat_ngram_size)
        candidate = candidates[0, begin_idx:-1]
        candidate = candidate.tolist()

        candidate_output = self.tokenizer.decode(candidate, skip_special_tokens=True)
        output = self.postprocess(text=candidate_output)
        if self.verbose:
            if conditions is not None: print("conditions:", conditions)
            print("greedy:", output)
        return output

    def infer_next_utterance_beam_search(self, utterances: List[str], speaker_ids: List[str], conditions: List[str],
                                         min_length: int, top_n: int, repetition_penalty: float, no_repeat_ngram_size: int,
                                         beam_size: int,
                                         prev_utterance: str, intersection_tolerance: float, max_retry: int, return_probs: bool):
        num_return_sequences = min(top_n * 4, beam_size)
        input_ids, token_type_ids = self.encode(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions, max_retry=max_retry)
        begin_idx = len(input_ids[0])
        candidates = self.model.generate(input_ids,
                                         min_length=min_length,
                                         max_length=self.timesteps,
                                         repetition_penalty=repetition_penalty,
                                         no_repeat_ngram_size=no_repeat_ngram_size,
                                         num_beams=beam_size,
                                         early_stopping=True,
                                         num_return_sequences=num_return_sequences)
        candidates = candidates[:, begin_idx:-1]
        candidates = candidates.tolist()

        probs = [-1] * len(candidates)
        output = self.decode(candidates=candidates, probs=probs, min_length=min_length, top_n=top_n, prev_utterance=prev_utterance, intersection_tolerance=intersection_tolerance)
        if self.verbose:
            if conditions is not None: print("conditions:", conditions)
            print("beam_search:\n", output)
        return output

    def infer_next_utterance_top_k_sampling(self, utterances: List[str], speaker_ids: List[str], conditions: List[str],
                                             min_length: int, top_n: int, repetition_penalty: float, no_repeat_ngram_size: int,
                                             top_k: float, top_p: float,
                                             prev_utterance: str, intersection_tolerance: float, max_retry: int, return_probs: bool):
        num_return_sequences = min(top_n * 4, top_k)
        input_ids, type_ids = self.encode(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions, max_retry=max_retry)
        begin_idx = len(input_ids[0])
        candidates = self.model.generate(input_ids,
                                         do_sample=True,
                                         min_length=min_length,
                                         max_length=self.timesteps,
                                         repetition_penalty=repetition_penalty,
                                         no_repeat_ngram_size=no_repeat_ngram_size,
                                         top_k=top_k,
                                         top_p=top_p,
                                         num_return_sequences=num_return_sequences)
        candidates = candidates[:, begin_idx:-1]
        candidates = candidates.tolist()

        probs = [-1] * len(candidates)
        output = self.decode(candidates=candidates, probs=probs, min_length=min_length, top_n=top_n, prev_utterance=prev_utterance, intersection_tolerance=intersection_tolerance)
        if self.verbose:
            if conditions is not None: print("conditions:", conditions)
            print("random_sampling:\n", output)
        return output

    def encode(self, utterances: List[str], speaker_ids: List[int], conditions: List[str], max_retry: int):
        utterances, speaker_ids = merge_utterances(utterances=utterances, speaker_ids=speaker_ids)
        input_ids = []
        token_type_ids = []

        def _encode_row(utterances, speaker_ids, conditions):
            input_ids = []
            token_type_ids = []

            # encode_context
            context_input_ids, context_token_type_ids = self.encoder.encode_context(tokenizer=self.tokenizer, utterances=utterances, speaker_ids=speaker_ids, context_token_id=self.context_token_id)
            input_ids += context_input_ids
            token_type_ids += context_token_type_ids

            if conditions is not None:
                # encode_condition
                condition_input_ids, condition_token_type_ids = self.encoder.encode_condition(tokenizer=self.tokenizer, conditions=conditions, condition_token_id=self.condition_token_id)
                input_ids += condition_input_ids
                token_type_ids += condition_token_type_ids

            input_ids = input_ids + [self.tokenizer.bos_token_id]
            _tail_token_type_id = token_type_ids[-1] if len(token_type_ids) > 0 else self.user_speaker_id
            token_type_ids = token_type_ids + [_tail_token_type_id]
            return input_ids, token_type_ids

        input_length_upper = int(self.timesteps * 0.75)
        for i in range(0, max_retry):
            input_ids, token_type_ids = _encode_row(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
            if len(input_ids) <= input_length_upper and len(token_type_ids) <= input_length_upper: break
            utterances = utterances[1:]
            speaker_ids = speaker_ids[1:]

        input_ids = convert_to_tensor([input_ids], device=self.device).to(torch.long)
        token_type_ids = convert_to_tensor([token_type_ids], device=self.device).to(torch.long)
        return input_ids, token_type_ids

