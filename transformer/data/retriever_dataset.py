import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from transformer.data.dataset import DatasetFromDir, StreamingDatasetFromDir
from transformer.data.generator_dataset import BlenderBotEncoder
from transformer.data.utils import simplify_speaker_ids
from transformer.utils.common import get_nth_index, convert_to_tensor, shuffle_dictionary_lists, is_cpu_device

class ElectraDatasetFromDir(DatasetFromDir, BlenderBotEncoder):
    def __init__(self, data_dir, tokenizer, timesteps, batch_size, device="cpu", nprocs=1, encoding="utf-8", extension="json"):
        if not data_dir.endswith("/"): data_dir += "/"
        DatasetFromDir.__init__(self=self, data_dir=data_dir, batch_size=batch_size, device=device, nprocs=nprocs, encoding=encoding, extension=extension)
        self.tokenizer = tokenizer
        self.left_token_type_id = 0
        self.right_token_type_id = 1
        self.timesteps = timesteps
        self.preprocess()

    def preprocess(self):
        output = []
        for row in tqdm(self.data, initial=0, total=len(self.data), desc="Preprocessing data"):
            utterances = row["utterances"]
            speaker_ids = row["speaker_ids"]
            conditions = None
            if "conditions" in row:
                conditions = row["conditions"]
            context_input, candidate_input = self.make_inputs(utterances=utterances, speaker_ids=speaker_ids, conditions=conditions)
            # src_input_batch = ["<s>It <mask> retriever. My <mask> cute </s>", ... ]
            # tgt_input_batch = ["</s><s>My dog is cute. It is a golden retriever", ...]
            # tgt_output_batch = ["<s>My dog is cute. It is a golden retriever</s>", ...]
            left_encoded = self.encode_left_row(context=context_input)
            right_encoded = self.encode_right_row(candidate=candidate_input)
            if left_encoded is None or right_encoded is None: continue
            left_row = left_encoded
            right_row = right_encoded

            output_row = dict()
            output_row["left_input_ids"] = left_row[0]
            output_row["left_token_type_ids"] = left_row[1]
            output_row["left_attention_mask"] = left_row[2]
            output_row["right_input_ids"] = right_row[0]
            output_row["right_token_type_ids"] = right_row[1]
            output_row["right_attention_mask"] = right_row[2]
            output.append(output_row)

        self.raw_data = self.data.copy()
        self.data = output
        self.data_size = len(output)

    def encode_left_row(self, context):
        left_input_ids = []
        left_turn_ids = []
        for _context, _speaker_id in zip(context["context"], context["speaker_ids"]):
            _tokens = self.tokenizer.tokenize(str(_context))
            _input_ids = self.tokenizer.convert_tokens_to_ids(tokens=_tokens)
            _turn_ids = len(_input_ids) * [self.left_token_type_id] # len(_input_ids) * [_speaker_id]
            left_input_ids += _input_ids
            left_turn_ids += _turn_ids

        # assert
        if len(left_input_ids) <= 0: return None
        left_input_ids = [self.tokenizer.cls_token_id] + left_input_ids + [self.tokenizer.sep_token_id]
        left_turn_ids = [left_turn_ids[0]] + left_turn_ids + [left_turn_ids[-1]]
        left_input_mask = [1] * len(left_input_ids)
        # assert
        if len(left_input_ids) > self.timesteps or len(left_turn_ids) > self.timesteps or len(left_input_mask) > self.timesteps: return None

        left_input_ids += (self.timesteps - len(left_input_ids)) * [self.tokenizer.pad_token_id]
        left_turn_ids += (self.timesteps - len(left_turn_ids)) * [self.tokenizer.pad_token_id]
        left_input_mask += (self.timesteps - len(left_input_mask)) * [0]
        return left_input_ids, left_turn_ids, left_input_mask

    def encode_right_row(self, candidate):
        right_input_ids = []
        right_turn_ids = []
        for _candidate, _speaker_id in zip(candidate["candidate"], candidate["speaker_ids"]):
            _turn_id = self.right_token_type_id # _speaker_id
            _tokens = self.tokenizer.tokenize(str(_candidate))
            _input_ids = self.tokenizer.convert_tokens_to_ids(tokens=_tokens)
            _turn_ids = len(_input_ids) * [_turn_id]
            right_input_ids += _input_ids
            right_turn_ids += _turn_ids

        # assert
        if len(right_input_ids) <= 0: return None
        right_input_ids = [self.tokenizer.cls_token_id] + right_input_ids + [self.tokenizer.sep_token_id]
        right_turn_ids = [right_turn_ids[0]] + right_turn_ids + [right_turn_ids[-1]]
        right_input_mask = [1] * len(right_input_ids)
        # assert
        if len(right_input_ids) > self.timesteps or len(right_turn_ids) > self.timesteps or len(right_input_mask) > self.timesteps: return None

        right_input_ids += (self.timesteps - len(right_input_ids)) * [self.tokenizer.pad_token_id]
        right_turn_ids += (self.timesteps - len(right_turn_ids)) * [self.tokenizer.pad_token_id]
        right_input_mask += (self.timesteps - len(right_input_mask)) * [0]
        return right_input_ids, right_turn_ids, right_input_mask


class RetrieverDataLoader(DataLoader):
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
        context_batch_size = len(inputs["left_input_ids"])
        candidate_batch_size = len(inputs["right_input_ids"])
        inputs["labels"] = list(range(0, context_batch_size))
        return inputs

    def get_batch(self):
        return next(self.__iter__())

    def check(self, row_idx=0):
        batch = self.get_batch()
        headers = []
        nth_rows = []
        for k, v in batch.items():
            if k.endswith("labels"): continue
            headers.append(k)
            nth_rows.append(batch[k][row_idx])
            if k.endswith("input_ids"):
                row = [_id for _id in batch[k][row_idx] if _id >= 0 and _id != self.dataset.tokenizer.pad_token_id]
                decoded = self.dataset.tokenizer.decode(row, skip_special_tokens=False)
                print("{}:\t{}".format(k, decoded))
        print()
        print("header:", headers)
        print(np.stack(nth_rows, axis=-1))