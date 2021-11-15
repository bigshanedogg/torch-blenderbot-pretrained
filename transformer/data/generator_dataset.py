import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from transformer.data.dataset import DatasetFromDir, StreamingDatasetFromDir
from transformer.data.utils import simplify_speaker_ids
from transformer.utils.common import get_nth_index, convert_to_tensor, shuffle_dictionary_lists, is_cpu_device

class BlenderBotEncoder:
    context_token_id = None
    condition_token_id = None
    model_speaker_id = 0
    user_speaker_id = 1

    def make_inputs(self, utterances, speaker_ids, conditions: List[List[str]]):
        # split sequence into context and candidate
        # context: previous dialogue history
        # candidate: next_utterance
        # condition: one of (Persona, Topic/Wow, Candidate)
        # context_input_row: {"context": [context_utterance_1, context_utterance_2, ...], "condition":[condition_str_1, condition_str_2, ...], "turn":[turn_id_1, turn_id_2, ...]}
        # candidate_input_row: {"candidate": [candidate_utterance], "turn":[turn_id_1, turn_id_2, ...]}
        speaker_ids = simplify_speaker_ids(speaker_ids=speaker_ids, user_id=self.user_speaker_id, model_id=self.model_speaker_id)
        context_input_row = self.extract_context_input_row(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
        candidate_input_row = self.extract_candidate_input_row(utterances=utterances, speaker_ids=speaker_ids)
        return context_input_row, candidate_input_row

    def extract_context_input_row(self, utterances, speaker_ids, conditions):
        last_user_utterance_idx = get_nth_index(obj=speaker_ids, value=self.user_speaker_id, n=-1)
        context_utterances = utterances[:last_user_utterance_idx + 1]
        context_speaker_ids = speaker_ids[:last_user_utterance_idx + 1]

        context_input_row = dict()
        context_input_row["context"] = context_utterances
        context_input_row["speaker_ids"] = context_speaker_ids
        context_input_row["conditions"] = conditions
        return context_input_row

    def extract_candidate_input_row(self, utterances, speaker_ids):
        last_user_utterance_idx = get_nth_index(obj=speaker_ids, value=self.user_speaker_id, n=-1)
        candidate_utterances = utterances[last_user_utterance_idx + 1:]
        candidate_speaker_ids = speaker_ids[last_user_utterance_idx + 1:]

        candidate_input_row = dict()
        candidate_input_row["candidate"] = candidate_utterances
        candidate_input_row["speaker_ids"] = candidate_speaker_ids
        return candidate_input_row

    def encode_context(self, tokenizer, utterances, speaker_ids, context_token_id):
        input_ids = []
        turn_ids = []
        for _context_utterance, _speaker_id in zip(utterances, speaker_ids):
            _turn_id = _speaker_id
            _input_ids = tokenizer.encode(str(_context_utterance))
            _turn_ids = len(_input_ids) * [_turn_id]
            input_ids += _input_ids
            turn_ids += _turn_ids
        if context_token_id is not None:
            input_ids = [context_token_id] + input_ids
            _turn_id = turn_ids[0] if len(turn_ids) > 0 else self.user_speaker_id
            turn_ids = [_turn_id] + turn_ids
        return input_ids, turn_ids

    def encode_condition(self, tokenizer, conditions, condition_token_id):
        input_ids = []
        turn_ids = []
        for _condition in conditions:
            _input_ids = tokenizer.encode(_condition)
            _turn_ids = len(_input_ids) * [self.model_speaker_id]
            input_ids += _input_ids
            turn_ids += _turn_ids
        if condition_token_id is not None:
            input_ids = [condition_token_id] + input_ids
            _turn_id = turn_ids[0] if len(turn_ids) > 0 else self.model_speaker_id
            turn_ids = [_turn_id] + turn_ids
        return input_ids, turn_ids

    def encode_candidate(self, tokenizer, utterances, speaker_ids):
        if speaker_ids is None: speaker_ids = [self.model_speaker_id] * len(utterances)

        input_ids = []
        turn_ids = []
        labels = []
        for _candidate_utterance, _speaker_id in zip(utterances, speaker_ids):
            _turn_id = self.model_speaker_id
            _input_ids = tokenizer.encode(str(_candidate_utterance))
            _turn_ids = len(_input_ids) * [_turn_id]
            input_ids += _input_ids
            turn_ids += _turn_ids
            labels += _input_ids
        return input_ids, turn_ids, labels

class GptDatasetFromDir(DatasetFromDir, BlenderBotEncoder):
    def __init__(self, data_dir, tokenizer, timesteps, batch_size, device="cpu", nprocs=1, use_condition: bool = False, alpha_blending: float = -1, decode_margin: float = 0.15, encoding="utf-8", extension="json"):
        if not data_dir.endswith("/"): data_dir += "/"
        DatasetFromDir.__init__(self=self, data_dir=data_dir, batch_size=batch_size, device=device, nprocs=nprocs, encoding=encoding, extension=extension)
        self.use_condition = use_condition
        self.alpha_blending = alpha_blending if use_condition else -1
        if self.use_condition and "context_token_id" in tokenizer.__dict__:
            self.context_token_id = tokenizer.context_token_id
        if self.use_condition and "condition_token_id" in tokenizer.__dict__:
            self.condition_token_id = tokenizer.condition_token_id
        self.decode_margin = decode_margin
        self.tokenizer = tokenizer
        self.timesteps = timesteps
        self.preprocess()

    def preprocess(self):
        output = []
        for row in tqdm(self.data, initial=0, total=len(self.data), desc="Preprocessing data"):
            utterances = row["utterances"]
            speaker_ids = row["speaker_ids"]
            conditions = None
            context_input, candidate_input = self.make_inputs(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
            if "conditions" in row:
                if np.random.rand() < self.alpha_blending: conditions = [" ".join(candidate_input["candidate"])]
                else: conditions = row["conditions"]
            encoded = self.encode_row(context_utterances=context_input["context"], context_speaker_ids=context_input["speaker_ids"], conditions=conditions,
                                      candidate_utterances=candidate_input["candidate"], candidate_speaker_ids=candidate_input["speaker_ids"])
            if encoded is None: continue
            row, labels = encoded

            output_row = dict()
            output_row["input_ids"] = row[0]
            output_row["token_type_ids"] = row[1]
            output_row["attention_mask"] = row[2]
            output_row["labels"] = labels
            output.append(output_row)

        self.raw_data = self.data.copy()
        self.data = output
        self.data_size = len(output)

    def encode_row(self, context_utterances: List[str], context_speaker_ids: List[int], conditions: List[str],
                   candidate_utterances: List[str], candidate_speaker_ids: List[int]):
        input_ids = []
        turn_ids = []
        labels = []

        # encode_context
        context_input_ids, context_turn_ids = self.encode_context(tokenizer=self.tokenizer, utterances=context_utterances, speaker_ids=context_speaker_ids, context_token_id=self.context_token_id)
        input_ids += context_input_ids
        turn_ids += context_turn_ids

        if self.use_condition and conditions is not None:
            # encode_condition
            condition_input_ids, condition_turn_ids = self.encode_condition(tokenizer=self.tokenizer, conditions=conditions, condition_token_id=self.condition_token_id)
            input_ids += condition_input_ids
            turn_ids += condition_turn_ids

        # assert
        if len(input_ids) <= 0: return None

        # add special_tokens
        labels += len(input_ids) * [-100]
        input_ids = input_ids + [self.tokenizer.bos_token_id]
        _tail_turn_id = turn_ids[-1] if len(turn_ids) > 0 else self.user_speaker_id
        turn_ids = turn_ids + [_tail_turn_id]
        labels = labels + [self.tokenizer.bos_token_id]

        # encode_candidate
        candidate_input_ids, candidate_turn_ids, candidate_labels = self.encode_candidate(tokenizer=self.tokenizer, utterances=candidate_utterances, speaker_ids=candidate_speaker_ids)
        input_ids += candidate_input_ids
        turn_ids += candidate_turn_ids
        labels += candidate_labels

        # add special_tokens
        input_ids = input_ids + [self.tokenizer.eos_token_id]
        turn_ids = turn_ids + [self.model_speaker_id]
        attention_mask = [1] * len(input_ids)
        labels = labels + [self.tokenizer.eos_token_id]

        upperbound = int(self.timesteps * (1 - self.decode_margin))
        if len(input_ids) > upperbound or len(turn_ids) > upperbound or len(labels) > upperbound: return None

        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.timesteps - len(input_ids))
        turn_ids = turn_ids + [self.tokenizer.pad_token_id] * (self.timesteps - len(turn_ids))
        attention_mask = attention_mask + [0] * (self.timesteps - len(attention_mask))
        labels = labels + [-100] * (self.timesteps - len(labels))
        return (input_ids, turn_ids, attention_mask), labels

class BartDatasetFromDir(DatasetFromDir, BlenderBotEncoder):
    def __init__(self, data_dir, tokenizer, timesteps, batch_size, device="cpu", nprocs=1, use_condition: bool = False, alpha_blending: float = -1, encoding="utf-8", extension="json"):
        if not data_dir.endswith("/"): data_dir += "/"
        DatasetFromDir.__init__(self=self, data_dir=data_dir, batch_size=batch_size, device=device, nprocs=nprocs, encoding=encoding, extension=extension)
        self.use_condition = use_condition
        self.alpha_blending = alpha_blending if use_condition else -1
        if self.use_condition and "context_token_id" in tokenizer.__dict__:
            self.context_token_id = tokenizer.context_token_id
        if self.use_condition and "condition_token_id" in tokenizer.__dict__:
            self.condition_token_id = tokenizer.condition_token_id
        self.tokenizer = tokenizer
        self.timesteps = timesteps
        self.preprocess()

    def preprocess(self):
        output = []
        for row in tqdm(self.data, initial=0, total=len(self.data), desc="Preprocessing data"):
            utterances = row["utterances"]
            speaker_ids = row["speaker_ids"]
            conditions = None
            context_input, candidate_input = self.make_inputs(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
            if "conditions" in row:
                if np.random.rand() < self.alpha_blending: conditions = [" ".join(candidate_input["candidate"])]
                else: conditions = row["conditions"]
            # src_input_batch = ["<s>It <mask> retriever. My <mask> cute </s>", ... ]
            # tgt_input_batch = ["</s><s>My dog is cute. It is a golden retriever", ...]
            # tgt_output_batch = ["<s>My dog is cute. It is a golden retriever</s>", ...]
            src_encoded = self.encode_src_row(utterances=context_input["context"], speaker_ids=context_input["speaker_ids"], conditions=conditions)
            tgt_encoded = self.encode_tgt_row(utterances=candidate_input["candidate"], speaker_ids=candidate_input["speaker_ids"])
            if src_encoded is None or tgt_encoded is None: continue
            src_row = src_encoded
            tgt_row, labels = tgt_encoded

            output_row = dict()
            output_row["input_ids"] = src_row[0]
            output_row["token_type_ids"] = src_row[1]
            output_row["attention_mask"] = src_row[2]
            output_row["decoder_input_ids"] = tgt_row[0]
            output_row["decoder_token_type_ids"] = tgt_row[1]
            output_row["decoder_attention_mask"] = tgt_row[2]
            output_row["labels"] = labels
            output.append(output_row)

        self.raw_data = self.data.copy()
        self.data = output
        self.data_size = len(output)

    def encode_src_row(self, utterances: List[str], speaker_ids: List[int], conditions: List[str]):
        # src_input_batch = ["<s>It <mask> retriever. My <mask> cute </s>", ... ]
        src_input_ids = []
        src_turn_ids = []

        # encode_context
        context_input_ids, context_turn_ids = self.encode_context(tokenizer=self.tokenizer, utterances=utterances, speaker_ids=speaker_ids, context_token_id=self.context_token_id)
        src_input_ids += context_input_ids
        src_turn_ids += context_turn_ids

        if self.use_condition and conditions is not None:
            # encode_condition
            condition_input_ids, condition_turn_ids = self.encode_condition(tokenizer=self.tokenizer, conditions=conditions, condition_token_id=self.condition_token_id)
            src_input_ids += condition_input_ids
            src_turn_ids += condition_turn_ids

        # assert
        if len(src_input_ids) <= 0: return None

        src_input_ids = [self.tokenizer.bos_token_id] + src_input_ids + [self.tokenizer.eos_token_id]
        _head_src_turn_id = src_turn_ids[0] if len(src_turn_ids) > 0 else self.user_speaker_id
        _tail_src_turn_id = src_turn_ids[-1] if len(src_turn_ids) > 0 else self.user_speaker_id
        src_turn_ids = [_head_src_turn_id] + src_turn_ids + [_tail_src_turn_id]
        src_input_mask = [1] * len(src_input_ids)

        # assert
        if len(src_input_ids) > self.timesteps or len(src_turn_ids) > self.timesteps or len(src_input_mask) > self.timesteps: return None

        # add special_tokens
        src_input_ids += (self.timesteps - len(src_input_ids)) * [self.tokenizer.pad_token_id]
        src_turn_ids += (self.timesteps - len(src_turn_ids)) * [self.tokenizer.pad_token_id]
        src_input_mask += (self.timesteps - len(src_input_mask)) * [0]
        return src_input_ids, src_turn_ids, src_input_mask

    def encode_tgt_row(self, utterances: List[str], speaker_ids: List[int]):
        # tgt_input_batch = ["</s><s>My dog is cute. It is a golden retriever", ...]
        # tgt_output_batch = ["<s>My dog is cute. It is a golden retriever</s>", ...]
        tgt_input_ids = []
        tgt_turn_ids = []
        labels = []

        # encode_candidate
        candidate_input_ids, candidate_turn_ids, candidate_labels = self.encode_candidate(tokenizer=self.tokenizer, utterances=utterances, speaker_ids=speaker_ids)
        tgt_input_ids += candidate_input_ids
        tgt_turn_ids += candidate_turn_ids
        labels += candidate_labels

        # assert
        if len(tgt_input_ids) <= 0: return None

        # add special_tokens
        tgt_input_ids = [self.tokenizer.eos_token_id] + [self.tokenizer.bos_token_id] + tgt_input_ids
        _head_tgt_turn_id = tgt_turn_ids[0] if len(tgt_turn_ids) > 0 else self.model_speaker_id
        tgt_turn_ids = [_head_tgt_turn_id] + [_head_tgt_turn_id] + tgt_turn_ids
        labels = [self.tokenizer.bos_token_id] + labels + [self.tokenizer.eos_token_id]
        tgt_input_mask = [1] * len(tgt_input_ids)

        # assert
        if len(tgt_input_ids) > self.timesteps or len(tgt_turn_ids) > self.timesteps or len(labels) > self.timesteps or len(tgt_input_mask) > self.timesteps: return None

        tgt_input_ids += (self.timesteps - len(tgt_input_ids)) * [self.tokenizer.pad_token_id]
        tgt_turn_ids += (self.timesteps - len(tgt_turn_ids)) * [self.tokenizer.pad_token_id]
        labels += (self.timesteps - len(labels)) * [-100]
        tgt_input_mask += (self.timesteps - len(tgt_input_mask)) * [0]
        return (tgt_input_ids, tgt_turn_ids, tgt_input_mask), labels

class GeneratorDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, device):
        num_workers = 1
        pin_memory = False
        if not is_cpu_device(device):
            num_workers = 2
            pin_memory = True
        DataLoader.__init__(self=self, dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.device = device
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch, shuffle=True):
        inputs = dict()
        for row in batch:
            for k, v in row.items():
                if k not in inputs: inputs[k] = []
                inputs[k].append(v)

        # shuffle
        if shuffle: inputs = shuffle_dictionary_lists(dictionaries=[inputs])[0]
        return inputs

    def get_batch(self):
        return next(self.__iter__())

    def check(self, row_idx=0):
        batch = self.get_batch()
        headers = []
        nth_rows = []
        for k, v in batch.items():
            headers.append(k)
            nth_rows.append(batch[k][row_idx])
            if k.endswith("input_ids") or k.endswith("labels"):
                row = [_id for _id in batch[k][row_idx] if _id >= 0 and _id != self.dataset.tokenizer.pad_token_id]
                decoded = self.dataset.tokenizer.decode(row, skip_special_tokens=False)
                print("{}:\t{}".format(k, decoded))
        print()
        print("header:", headers)
        print(np.stack(nth_rows, axis=-1))