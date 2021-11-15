from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel
from microservices.models.interface import Request

class InferNextUtteranceGreedyRequest(Request):
    utterances: List[str]
    speaker_ids: List[int]
    conditions: List[str] = None
    min_length: int = 10
    repetition_penalty: float = 2.0
    no_repeat_ngram_size: int = 3
    prev_utterance: str = None
    intersection_tolerance: float = 0.9
    max_retry: int = 5
    return_probs: bool = True

class InferNextUtteranceBeamSearchRequest(Request):
    utterances: List[str]
    speaker_ids: List[int]
    conditions: List[str] = None
    min_length: int = 10
    top_n: int = 5
    repetition_penalty: float = 2.0
    no_repeat_ngram_size: int = 3
    beam_size: int = 5
    prev_utterance: str = None
    intersection_tolerance: float = 0.9
    max_retry: int = 5
    return_probs: bool = True

class InferNextUtteranceTopKSamplingRequest(Request):
    utterances: List[str]
    speaker_ids: List[int]
    conditions: List[str] = None
    min_length: int = 10
    top_n: int = 5
    repetition_penalty: float = 2.0
    no_repeat_ngram_size: int = 3
    top_k: int = 40
    top_p: float = 0.95
    prev_utterance: str = None
    intersection_tolerance: float = 0.9
    max_retry: int = 5
    return_probs: bool = True

