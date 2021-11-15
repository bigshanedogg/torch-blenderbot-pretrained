from collections import Counter
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from transformer.utils.common import convert_to_tensor, convert_to_numpy

def get_language_modeling_accuracy(predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100):
    '''
    :param predictions: (batch_size, timesteps, vocab_size)
    :param targets: (batch_size, timesteps)
    :return:
    '''
    # predictions: (batch_size * timesteps, vocab_size)
    predictions = predictions.view(-1, predictions.size(-1))
    # predictions: (batch_size * timesteps, )
    predictions = torch.argmax(predictions, axis=-1)
    # targets: (batch_size * timesteps, )
    targets = targets.contiguous().view(-1)

    label_weights = None
    if ignore_index is not None:
        label_weights = (targets != ignore_index).float()
    else:
        label_weights = torch.ones_like(targets)

    numerator = torch.sum(label_weights * (predictions == targets).float())
    denominator = torch.sum(label_weights) + 1e-5
    accuracy = numerator / denominator
    return accuracy

def get_classification_accuracy(predictions: torch.Tensor, targets: torch.Tensor):
    '''
    :param predictions: (batch_size, num_class)
    :param targets: (batch_size, )
    :return:
    '''
    # predictions: (batch_size, )
    predictions = torch.argmax(predictions, axis=-1)
    label_weights = torch.ones_like(targets)
    numerator = torch.sum((predictions == targets).float())
    denominator = torch.sum(label_weights) + 1e-5
    accuracy = numerator / denominator
    return accuracy


def get_perplexity(predictions: torch.Tensor, targets: torch.Tensor, is_log_prob: bool, ignore_index: int = -100):
    '''
    :param predictions: (batch_size, timesteps, vocab_size)
    :param targets: (batch_size, timesteps)
    :return:
    '''
    # predictions: (batch_size * timesteps, vocab_size)
    batch_size = predictions.shape[0]
    predictions = predictions.view(-1, predictions.size(-1))
    # targets: (batch_size * timesteps, )
    targets = targets.contiguous().view(-1)

    label_weights = None
    if ignore_index is not None:
        label_weights = (targets != ignore_index).float()
    else:
        label_weights = torch.ones_like(targets)
    n = torch.sum(label_weights.reshape(batch_size, -1), axis=-1)

    p_w = predictions.gather(1, targets.unsqueeze(1)).squeeze()
    p_w = p_w * label_weights
    p_w = torch.sum(p_w.reshape(batch_size, -1), axis=-1)
    batch_perplexities = -1 / n * p_w
    if is_log_prob: batch_perplexities = torch.exp(batch_perplexities)
    perplexity = torch.mean(batch_perplexities)
    return perplexity


def get_bleu_score(predictions: torch.Tensor, targets: torch.Tensor, pad_token_id: int, eos_token_id: int, ngram: int = 4):
    '''
    :param predictions: (batch_size, timesteps, vocab_size)
    :param targets: (batch_size, timesteps)
    :return:
    '''
    # predictions, targets: (batch_size, timesteps)
    predictions = convert_to_numpy(tensor=predictions)
    predictions = np.argmax(predictions, axis=-1).tolist()
    targets = convert_to_numpy(tensor=targets).tolist()

    if pad_token_id in targets[0]:
        eos_token_id = targets[0][targets[0].index(pad_token_id) - 1]
    else:
        eos_token_id = targets[0][-1]
    bleu_score = calculate_bleu(predictions=predictions, targets=targets, ngram=ngram, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
    return bleu_score

def calculate_bleu(predictions, targets, ngram, pad_token_id, eos_token_id, weights: List[float] = None, clipping: bool = True):
    bleu_score = 0
    for _prediction, _target in zip(predictions, targets):
        prediction = []
        for col in _prediction:
            prediction.append(col)
            if col == eos_token_id: break

        target = []
        for col in _target:
            target.append(col)
            if col == pad_token_id or col == eos_token_id: break

        _bleu_score = _calculate_bleu_row(prediction=prediction, target=target, ngram=ngram, weights=weights, clipping=clipping)
        bleu_score += _bleu_score

    bleu_score = bleu_score / len(predictions)
    return bleu_score

def _calculate_bleu_row(prediction, target, ngram, weights: List[float] = None, clipping:bool = True, underflow: float = 1e-7):
    sum_log_p_n = 0.0
    weight = 1/ngram
    for n in range(1, ngram+1):
        prediction_ngram = _split_into_ngrams(row=prediction, ngram=n)
        prediction_counter = Counter(prediction_ngram)
        target_ngram = _split_into_ngrams(row=target, ngram=n)
        target_counter = Counter(target_ngram)
        if weights is not None: weight = weights[n-1]

        denominator = len(prediction_ngram)
        numerator = 0
        for p_ngram in prediction_ngram:
            if p_ngram in target_ngram:
                if p_ngram not in prediction_counter: continue
                if clipping: numerator += min(prediction_counter[p_ngram], target_counter[p_ngram])
                else: numerator += prediction_counter[p_ngram]
        p_n = numerator / (denominator + underflow) # prevent divided by zero
        log_p_n = np.log((p_n + underflow)) # prevent divided by zero
        sum_log_p_n += (weight * log_p_n)

    brevity_penalty = min(0, (1 - len(target)/(len(prediction)+underflow)))
    log_blue_score = brevity_penalty + sum_log_p_n
    bleu_score = np.exp(log_blue_score)
    return bleu_score

def _split_into_ngrams(row, ngram):
    output = []
    for i in range(0, len(row)-ngram):
        _ngram = row[i:i+ngram]
        output.append(tuple(_ngram))
    return output