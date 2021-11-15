from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel
from microservices.models.interface import Request

class SetDeviceRequest(Request):
    device: str

class LoadModelRequest(Request):
    path: str

class InferNextUtteranceRequest(Request):
    utterances: List[str]
    speaker_ids: List[int]
    min_length: int = 10
    top_n: int = 5
    weight_bm25: bool = False
    prev_utterance: str = None
    intersection_tolerance: float = 0.9
    max_retry: int = 5

class InferNextUtteranceBM25Request(Request):
    utterances: List[str]
    min_length: int = 10
    top_n: int = 5
    prev_utterance: str = None
    intersection_tolerance: float = 0.9