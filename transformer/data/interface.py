import json
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
from transformer.utils.tokenizer import MecabTokenizer, SpmTokenizer
from transformer.assertions.object_assertion import DataAssertion
from transformer.data.utils import get_iter_range
from transformer.utils.common import show_five_nums

class DatasetInterface(DataAssertion):
    def __init__(self, batch_size, device="cpu", nprocs=1):
        self.batch_size = batch_size
        self.device = device
        self.nprocs = nprocs
        self.set_iter_range()
        self.iter_range_update = False

    def shuffle_data(self, data):
        data = random.sample(data, len(data))
        return data

    def count_lines(self, file_path):
        self.assert_implemented(method_name="count_lines")

    def get_all_data(self):
        self.assert_implemented(method_name="get_all_data")

    def get_file_path_list(self, data_dir, extension):
        file_path_list = []
        if data_dir is not None:
            for filename in os.listdir(data_dir):
                if not filename.endswith(extension): continue
                file_path = data_dir + filename
                file_path_list.append(file_path)
        return file_path_list

    def parse_file(self, path):
        if self.extension == "txt":
            with open(path, "r", encoding=self.encoding) as fp:
                rows = []
                for row in fp:
                    row = row.strip()
                    rows.append(row)
                    if len(rows) >= self.batch_size:
                        yield from rows
                        rows = []
        elif self.extension == "json":
            data = []
            with open(path, "r", encoding=self.encoding) as fp:
                data = json.load(fp)
            rows = []
            for row in data:
                rows.append(row)
                if len(rows) >= self.batch_size:
                    yield from rows
                    rows = []

    def set_iter_range(self):
        # if worker info is None: single-process data loading, return the full iterator
        # else: multi-process data loading, return the splited iterator
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.iter_start, self.iter_end = 0, self.data_size
        else:
            self.iter_start, self.iter_end = get_iter_range(worker_id=worker_info.id,
                                                            num_workers=worker_info.num_workers,
                                                            device=self.device, nprocs=self.nprocs, data_size=self.data_size)
        self.data_size = self.iter_end - self.iter_start
        self.iter_range_update = True

class DataLoaderInterface(DataLoader, DataAssertion):
    def __init__(self, dataset, preprocessor, batch_size, device, nprocs, num_workers, pin_memory):
        self.preprocessor = preprocessor
        self.device = device
        self.nprocs = nprocs

        DataLoader.__init__(self=self, dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.set_iter_range(num_workers=self.num_workers, device=self.device, nprocs=self.nprocs, data_size=len(self.dataset), batch_size=self.batch_size)
        self.collate_fn = self._collate_fn

    def set_iter_range(self, num_workers, device, nprocs, data_size, batch_size):
        iter_start = 0
        iter_end = data_size
        if num_workers > 0:
            iter_start, _ = get_iter_range(worker_id=0, num_workers=num_workers, device=device, nprocs=nprocs,
                                           data_size=data_size)
            _, iter_end = get_iter_range(worker_id=num_workers - 1, num_workers=num_workers, device=device,
                                         nprocs=nprocs, data_size=data_size)
        self.iter_start = int(np.ceil(iter_start / batch_size))
        self.iter_end = int(np.ceil(iter_end / batch_size))

    def get_batch(self):
        return next(self.__iter__())

    def summary(self):
        self.assert_implemented(method_name="summary")

    def _summary(self, rows: List[str], sentence_to_ids_func, verbose: bool = True):
        # mecab_tokenizer = MecabTokenizer()
        # spm_tokenizer = SpmTokenizer(mlm_ratio=0.0, random_mask_ratio=0.0, skip_mask_ratio=0.0)
        # spm_tokenizer.load_spm_model(path=spm_model_path)
        #
        # tokenize_func = None
        # if language == "eng":
        #     tokenize_func = mecab_tokenizer.tokenize_eng
        # elif language == "kor":
        #     tokenize_func = mecab_tokenizer.tokenize_kor

        length_list = []
        row_iter = tqdm(rows, initial=0, total=len(rows), desc="Extracting length_list")
        for row in row_iter:
            if isinstance(row, list):
                row = [str(element) for element in row]
                row = " ".join(row)
            elif isinstance(row, int):
                row = str(row)
            elif isinstance(row, float):
                row = str(int(row))

            token_ids = sentence_to_ids_func(sentence=row, mask=False)
            # tokens = tokenize_func(sentence=row, return_pos=False)
            # token_ids = spm_tokenizer.tokens_to_ids(tokens=tokens, mask=False)
            length_list.append(len(token_ids))
        output = show_five_nums(data=length_list, verbose=verbose)
        return output