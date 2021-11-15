from collections import Counter
from typing import List, Tuple, Dict
import numpy as np

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

        _bleu_score = calculate_bleu_row(prediction=prediction, target=target, ngram=ngram, weights=weights, clipping=clipping)
        bleu_score += _bleu_score

    bleu_score = bleu_score / len(predictions)
    return bleu_score

def calculate_bleu_row(prediction, target, ngram, weights: List[float] = None, clipping:bool = True, underflow: float = 1e-7):
    sum_log_p_n = 0.0
    weight = 1/ngram
    for n in range(1, ngram+1):
        prediction_ngram = split_into_ngrams(row=prediction, ngram=n)
        prediction_counter = Counter(prediction_ngram)
        target_ngram = split_into_ngrams(row=target, ngram=n)
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

def split_into_ngrams(row, ngram):
    output = []
    for i in range(0, len(row)-ngram):
        _ngram = row[i:i+ngram]
        output.append(tuple(_ngram))
    return output