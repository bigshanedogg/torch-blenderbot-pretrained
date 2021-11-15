import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from transformer.data.generator_dataset import BlenderBotEncoder
from transformer.data.utils import merge_utterances
from transformer.models.poly_encoder import PolyEncoder
from transformer.models.utils import ModelFilenameConstants, load_tokenizer, load_state_dict, load_candidates
from transformer.utils.common import convert_to_tensor, convert_to_numpy
from transformer.utils.information_retrieval import BM25Okapi
from transformer.services.interface import ServiceInterface

class PolyEncoderDialogRetriever(ServiceInterface):
    print("loading candidates...")
    def set_candidates(self, candidates, candidate_embeds):
        candidates, candidate_embeds = self.remove_dumplicate_candidates(candidates=candidates, candidate_embeds=candidate_embeds)
        self.candidates = np.array(candidates)
        self.candidate_embeds = convert_to_tensor(data=candidate_embeds, device=self.device)
        print("loading bm25...")
        self.bm25.set_candidates(candidates=self.candidates)

    def load_model(self, model_dir, encoder_type: str = "electra"):
        if not model_dir.endswith("/"): model_dir += "/"
        self.encoder_type = encoder_type
        self.m_code = 64
        self.timesteps = 128
        self.encoder = BlenderBotEncoder()
        self.left_token_type_id = 0
        self.right_token_type_id = 1

        print("loading tokenizer...")
        self.tokenizer = load_tokenizer(path=model_dir, model_type=self.encoder_type)
        self.vocab_size = len(self.tokenizer)
        print("loading model...")
        self.model = PolyEncoder(encoder_type=self.encoder_type, vocab_size=self.vocab_size, m_code=self.m_code)
        self.model = self.model.to(self.device)
        print("loading model_state_dict...")
        self.model = load_state_dict(object=self.model, path=model_dir)
        print("loading candidates...")
        data = load_candidates(path=model_dir)
        candidates, candidate_embeds = data
        candidates, candidate_embeds = self.remove_dumplicate_candidates(candidates=candidates, candidate_embeds=candidate_embeds)
        self.candidates = np.array(candidates)
        self.candidate_embeds = convert_to_tensor(data=candidate_embeds, device=self.device)
        print("loading bm25...")
        self.bm25 = BM25Okapi(tokenizer=self.tokenizer, dataset=None)
        self.bm25.set_candidates(candidates=self.candidates)
        return model_dir

    def infer_next_utterance(self, utterances: List[str], speaker_ids: List[str],
                                    min_length: int, top_n: int, weight_bm25: bool,
                                    prev_utterance: str, intersection_tolerance: float, max_retry: int):
        inputs = self.encode(utterances=utterances, speaker_ids=speaker_ids, max_retry=max_retry)
        outputs = self.model.forward_left(inputs=inputs, candidate_embeds=self.candidate_embeds)
        log_probs = convert_to_numpy(outputs["logits"][0])
        probs = np.exp(log_probs)

        if weight_bm25:
            context = " ".join(utterances)
            bm25_scores = self.bm25.get_scores(context=context, normalize=True)
            bm25_scores = np.array(bm25_scores)
            probs = probs * bm25_scores

        output = self.sort_and_rank(candidates=self.candidates, probs=probs, top_n=top_n, min_length=min_length, prev_utterance=prev_utterance, intersection_tolerance=intersection_tolerance)
        if self.verbose: print("sort_and_rank:\n", output)
        return output

    def infer_next_utterance_bm25(self, utterances: List[str],
                                  min_length: int, top_n: int,
                                  prev_utterance: str, intersection_tolerance: float):
        context = " ".join(utterances)
        probs = self.bm25.get_scores(context=context, normalize=True)
        probs = np.array(probs)
        output = self.sort_and_rank(candidates=self.bm25.candidates, probs=probs, top_n=top_n, min_length=min_length, prev_utterance=prev_utterance, intersection_tolerance=intersection_tolerance)
        if self.verbose: print("bm25:\n", output)
        return output

    def encode(self, utterances: List[str], speaker_ids: List[str], max_retry: int):
        utterances, speaker_ids = merge_utterances(utterances=utterances, speaker_ids=speaker_ids)
        left_input_ids = []
        left_token_type_ids = []
        left_attention_mask = []

        def _encode_left_row(utterances, speaker_ids):
            left_input_ids = []
            left_turn_ids = []

            for speaker_id, utterance in zip(speaker_ids, utterances):
                _tokens = self.tokenizer.tokenize(utterance)
                _input_ids = self.tokenizer.convert_tokens_to_ids(tokens=_tokens)
                _turn_ids = len(_input_ids) * [self.left_token_type_id]  # len(_input_ids) * [_speaker_id]
                left_input_ids += _input_ids
                left_turn_ids += _turn_ids

            left_input_ids = [self.tokenizer.cls_token_id] + left_input_ids + [self.tokenizer.sep_token_id]
            left_turn_ids = [left_turn_ids[0]] + left_turn_ids + [left_turn_ids[-1]]
            left_input_mask = [1] * len(left_input_ids)
            return left_input_ids, left_turn_ids, left_input_mask

        input_length_upper = int(self.timesteps * 1.0)
        for i in range(0, max_retry):
            left_input_ids, left_token_type_ids, left_attention_mask = _encode_left_row(utterances=utterances, speaker_ids=speaker_ids)
            if len(left_input_ids) <= input_length_upper and len(left_token_type_ids) <= input_length_upper and len(left_attention_mask) <= input_length_upper:
                left_input_ids += (self.timesteps - len(left_input_ids)) * [self.tokenizer.pad_token_id]
                left_token_type_ids += (self.timesteps - len(left_token_type_ids)) * [self.tokenizer.pad_token_id]
                left_attention_mask += (self.timesteps - len(left_attention_mask)) * [0]
                break
            utterances = utterances[1:]
            speaker_ids = speaker_ids[1:]

        inputs = dict()
        inputs["left_input_ids"] = left_input_ids
        inputs["left_token_type_ids"] = left_token_type_ids
        inputs["left_attention_mask"] = left_attention_mask
        inputs = {k:convert_to_tensor([v], device=self.device) for k,v in inputs.items()}
        return inputs