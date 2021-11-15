import os
import sys
from fastapi import APIRouter
from transformer.utils.common import set_seed
from microservices.models import interface
from microservices.models import dialog_retriever
from microservices.models import dialog_generator
from microservices.resources.persona_bot import model_path_config, dialog_generators, dialog_retrievers
from microservices.utils.decorators import response_decorator

router = APIRouter()

@router.get("/get-version-info", response_model=interface.Response)
@response_decorator
def get_version_info():
    output = model_path_config
    return output

@router.post("/{module}/{version}/set-device", response_model=interface.Response)
@response_decorator
def set_device(module: str, version: str, request: interface.SetDeviceRequest):
    model = None
    if module == "dialog-generator": model = dialog_generators[version]

    output = model.set_device(device=request.device)
    return output

@router.post("/{module}/{version}/load-model", response_model=interface.Response)
@response_decorator
def load_model(module: str, version: str, request: interface.LoadModelRequest):
    model = None
    if module == "dialog-generator": model = dialog_generators[version]

    output = model.load_model(model_dir=request.path)
    return output

@router.post("/dialog-generator/{version}/infer-next-utterance/greedy", response_model=interface.Response)
@response_decorator
def infer_next_utterance_greedy(version: str, request: dialog_generator.InferNextUtteranceGreedyRequest):
    model = dialog_generators[version]
    output = model.infer_next_utterance_greedy(utterances=request.utterances, speaker_ids=request.speaker_ids, conditions=request.conditions,
                                               min_length=request.min_length, repetition_penalty=request.repetition_penalty, no_repeat_ngram_size=request.no_repeat_ngram_size,
                                               prev_utterance=request.prev_utterance, intersection_tolerance=request.intersection_tolerance, max_retry=request.max_retry, return_probs=request.return_probs)
    return output

@router.post("/dialog-generator/{version}/infer-next-utterance/beam-search", response_model=interface.Response)
@response_decorator
def infer_next_utterance_beam_search(version: str, request: dialog_generator.InferNextUtteranceBeamSearchRequest):
    model = dialog_generators[version]
    output = model.infer_next_utterance_beam_search(utterances=request.utterances, speaker_ids=request.speaker_ids, conditions=request.conditions,
                                                    min_length=request.min_length, top_n=request.top_n, repetition_penalty=request.repetition_penalty, no_repeat_ngram_size=request.no_repeat_ngram_size,
                                                    beam_size=request.beam_size,
                                                    prev_utterance=request.prev_utterance, intersection_tolerance=request.intersection_tolerance, max_retry=request.max_retry, return_probs=request.return_probs)
    return output

@router.post("/dialog-generator/{version}/infer-next-utterance/top-k-sampling", response_model=interface.Response)
@response_decorator
def infer_next_utterance_top_k_sampling(version: str, request: dialog_generator.InferNextUtteranceTopKSamplingRequest):
    model = dialog_generators[version]
    output = model.infer_next_utterance_top_k_sampling(utterances=request.utterances, speaker_ids=request.speaker_ids, conditions=request.conditions,
                                                       min_length=request.min_length, top_n=request.top_n, repetition_penalty=request.repetition_penalty, no_repeat_ngram_size=request.no_repeat_ngram_size,
                                                       top_k=request.top_k, top_p=request.top_p,
                                                       prev_utterance=request.prev_utterance, intersection_tolerance=request.intersection_tolerance, max_retry=request.max_retry, return_probs=request.return_probs)
    return output

@router.post("/dialog-retriever/{version}/infer-next-utterance", response_model=interface.Response)
@response_decorator
def infer_next_utterance(version: str, request: dialog_retriever.InferNextUtteranceRequest):
    model = dialog_retrievers[version]
    output = model.infer_next_utterance(utterances=request.utterances, speaker_ids=request.speaker_ids,
                                                    min_length=request.min_length, top_n=request.top_n, weight_bm25=request.weight_bm25,
                                                    prev_utterance=request.prev_utterance, intersection_tolerance=request.intersection_tolerance,
                                                    max_retry=request.max_retry)
    return output

@router.post("/dialog-retriever/{version}/infer-next-utterance/bm25", response_model=interface.Response)
@response_decorator
def infer_next_utterance_bm25(version: str, request: dialog_retriever.InferNextUtteranceBM25Request):
    model = dialog_retrievers[version]
    output = model.infer_next_utterance_bm25(utterances=request.utterances,
                                                         min_length=request.min_length, top_n=request.top_n,
                                                         prev_utterance=request.prev_utterance, intersection_tolerance=request.intersection_tolerance)
    return output