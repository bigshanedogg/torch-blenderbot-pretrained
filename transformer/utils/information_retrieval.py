from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import rank_bm25
from KoBERTScore import BERTScore
from transformer.models.interface import ModelInterface
from transformer.assertions.object_assertion import ModelAssertion
from transformer.models.utils import compute_hits, compute_semantic_score
from transformer.utils.common import get_now_str, clean_text, get_last_index

class BM25Okapi(ModelInterface):
    def __init__(self, tokenizer, dataset, user_speaker_id: int = 1, temp_dir="./"):
        ModelInterface.__init__(self)

        self.user_speaker_id = user_speaker_id
        self.tokenizer = tokenizer
        if dataset is not None:
            candidates = self.extract_candidates(dataset=dataset, concat_candidates=True, additional_responses=None, verbose=True)
            self.set_candidates(candidates=candidates)
            self.bm25 = rank_bm25.BM25Okapi(corpus=self.tokenized_candidates)

    def set_candidates(self, candidates):
        self.candidates = candidates
        self.tokenized_candidates = [self.tokenizer.tokenize(candidate) for candidate in self.candidates]
        self.bm25 = rank_bm25.BM25Okapi(corpus=self.tokenized_candidates)

    def get_scores(self, context, normalize: bool = False, underflow: float = 1e-7) -> List[float]:
        tokenized_context = self.tokenizer.tokenize(context)
        scores = self.bm25.get_scores(tokenized_context)
        if normalize:
            max_v = max(scores)
            min_v = min(scores)
            scores = (scores - min_v) / (max_v - min_v +underflow)
        scores = scores.tolist()
        return scores

    def get_top_n(self, context, n: int = 10):
        tokenized_context = self.tokenizer.tokenize(context)
        outputs = self.bm25.get_top_n(tokenized_context, self.candidates, n)
        return outputs

    def get_metric_inputs(self, tokenizer, data_loader, device, hits_k: List[int], **kwargs):
        top_n = max(hits_k)
        candidates = []
        labels = []
        data_iter = tqdm(data_loader.dataset.raw_data, initial=0, total=len(data_loader.dataset.raw_data), desc="Computing scores")
        for row_idx, row in enumerate(data_iter):
            last_index = get_last_index(row["speaker_ids"], value=self.user_speaker_id)
            context = row["utterances"][:last_index + 1]
            context = " ".join(context)
            candidate = self.get_top_n(context=context, n=top_n)
            candidates.append(candidate)

            label = row["utterances"][last_index + 1:]
            label = " ".join(label)
            labels.append(label)
        return candidates, labels

    def _compute_scores(self, metrics, tokenizer, data_loader, device, **kwargs):
        return self._compute_discrimitive_scores(metrics, tokenizer, data_loader, device, **kwargs)

    def extract_candidates(self, dataset, additional_responses: List[str], concat_candidates: bool = True, verbose: bool = True):
        candidates = []
        if dataset is not None:
            extract_iter = dataset.raw_data
            if verbose: extract_iter = tqdm(dataset.raw_data, initial=0, total=len(dataset.raw_data), desc="Extracting responses")
            for row in extract_iter:
                utterances = row["utterances"]
                speaker_ids = row["speaker_ids"]
                conditions = None
                if "conditions" in row:
                    conditions = row["conditions"]
                _, candidate_input = dataset.make_inputs(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
                _candidate_input_utterances = candidate_input["candidate"]
                if concat_candidates: _candidate_input_utterances = [" ".join(candidate_input["candidate"])]
                candidate_input_utterances = []
                for _candidate_input_utterance in _candidate_input_utterances:
                    _candidate_input_utterance = clean_text(_candidate_input_utterance)
                    if len(_candidate_input_utterance) < 5 or len(_candidate_input_utterance.split()) < 3: continue
                    candidate_input_utterances.append(_candidate_input_utterance)
                    candidates += candidate_input_utterances

        if additional_responses is not None:
            for additional_response in additional_responses:
                _candidate_input = clean_text(text=additional_response)
                candidates.append(_candidate_input)

        print("Total {} candidates has been extracted".format(len(candidates)))
        return candidates