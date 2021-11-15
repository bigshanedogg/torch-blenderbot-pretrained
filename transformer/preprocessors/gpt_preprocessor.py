import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from tqdm import tqdm
from transformer.utils.logger import get_logger
from transformer.utils.common import get_last_index
from transformer.utils.tokenizer import SpmTokenizer, HuggingFaceTokenizer
from transformer.trainers.utils import ModelFilenameConstants
from transformer.preprocessors.interface import PreprocessorInterface, HuggingFacePreprocessorInterface

logger = get_logger(name=__name__)

class GptPreprocessor(HuggingFacePreprocessorInterface):
    tokenizer = None
    src_tokenizer = None
    tgt_tokenizer = None
    is_next_label = 1
    not_next_label = 0

    def __init__(self, language: str, tokenizer_path: str, embedding_dict: Dict[str, int], config_path: str = "./config/preprocessor_config.json", verbose=False):
        PreprocessorInterface.__init__(self, config_path=config_path, verbose=verbose)
        self.assert_isin_languages(language=language)
        self.language = language
        self.src_language = language
        self.tgt_language = language
        self.tokenizer = HuggingFaceTokenizer()
        self.src_tokenizer = self.tokenizer
        self.tgt_tokenizer = self.tokenizer
        self.tokenizer.load_model(path=tokenizer_path)
        self.embedding_dict = embedding_dict

    def encode_row(self, src_input_row: Dict[str, List[str]], tgt_input_row: Dict[str, List[str]], src_timesteps: int, tgt_timesteps: int,
                   src_sep_token_ids: List[str], approach: str = "ignore") -> Tuple[int, Tuple[List[int], List[int]]]:
        status = 0
        inputs = dict()
        outputs = dict()

        # input_row
        input_ids, output_ids = self.encode_input_row(src_input_row=src_input_row, tgt_input_row=tgt_input_row, src_sep_token_ids=src_sep_token_ids)
        for k, v in input_ids.items():
            inputs[k] = v
        for k, v in output_ids.items():
            outputs[k] = v

        self.assert_isequal_elements_length(data=list(inputs.values()))
        self.assert_isequal_elements_length(data=list(outputs.values()))
        src_lower_bound = sum([len(v) for v in src_sep_token_ids])
        if self.is_proper_length(ids=inputs["token"], upper_bound=src_timesteps, lower_bound=src_lower_bound):
            inputs["token"] = self.pad_row(ids=inputs["token"], timesteps=src_timesteps, padding_value=self.src_tokenizer.special_token_dict["pad"]["id"])
            for k, v in inputs.items():
                if k == "token": continue
                v = self.pad_row(ids=v, timesteps=src_timesteps, padding_value=self.src_tokenizer.special_token_dict["pad"]["id"])
                inputs[k] = v
            outputs["token"] = self.pad_row(ids=outputs["token"], timesteps=tgt_timesteps, padding_value=-100)
            outputs["segment"] = self.pad_row(ids=outputs["segment"], timesteps=tgt_timesteps, padding_value=0)
        else:
            if approach == "stop":
                status = 1
                return status, (None, None)
            elif approach == "ignore":
                status = 1
                return status, (None, None)
            elif approach == "truncate":
                status = 1
                inputs["token"] = self.truncate_over_length(ids=inputs["token"], upper_bound=src_timesteps)
                inputs["token"] = self.attach_token(ids=inputs["token"], append_head=None, append_tail=None)
                for k, v in inputs.items():
                    if k == "token": continue
                    v = self.truncate_over_length(ids=v, upper_bound=src_timesteps)
                    v = self.attach_token(ids=v, append_head=None, append_tail=None)
                    inputs[k] = v
                outputs["token"] = self.truncate_over_length(ids=outputs["token"], upper_bound=tgt_timesteps - 1)
                outputs["token"] = self.attach_token(ids=outputs["token"], append_head=None, append_tail=self.tgt_tokenizer.special_token_dict["eos"]["id"])
                outputs["segment"] = self.truncate_over_length(ids=outputs["segment"], upper_bound=tgt_timesteps)
                outputs["segment"] = self.attach_token(ids=outputs["segment"], append_head=None, append_tail=None)
        return status, (inputs, outputs)

    def encode_input_row(self):
        self.assert_implemented(method_name="encode_input_row")

    def encode(self, src_inputs, tgt_inputs, src_timesteps: int, tgt_timesteps: int, src_sep_tokens: List[List[str]], approach: str = "ignore") -> Tuple[Dict[str, List[List[int]]], Dict[str, List[List[int]]], Dict[str, List[List[int]]]]:
        '''
        approach: How to filter rows longer than given timesteps
        # ignore: exclude the over_length_row
        # truncate: truncate tokens(ids) longer than timesteps
        # stop: raise AssertionError
        '''
        src_inputs_rows = src_inputs.copy()
        tgt_inputs_rows = tgt_inputs.copy()
        self.assert_equal_length(a=src_inputs, b=tgt_inputs)
        src_sep_token_ids = self.get_sep_token_ids(sep_tokens=src_sep_tokens, num_segments=src_sep_tokens[0], tokenizer=self.src_tokenizer)

        inputs = dict()
        inputs["token"] = []
        for k, v in self.embedding_dict.items():
            inputs[k] = []
        outputs = dict()
        outputs["lm"] = []
        outputs["segment"] = []

        over_length_row_cnt = 0
        for row_idx, (src_input_row, tgt_input_row) in enumerate(zip(src_inputs_rows, tgt_inputs_rows)):
            status, (_inputs, _outputs) = self.encode_row(src_input_row=src_input_row, tgt_input_row=tgt_input_row, src_timesteps=src_timesteps, tgt_timesteps=tgt_timesteps, src_sep_token_ids=src_sep_token_ids, approach=approach)
            if status > 0:
                self._raise_approach_error(approach=approach, row_idx=row_idx)
                over_length_row_cnt += 1
                if approach == "ignore": continue

            for k,v in _inputs.items():
                if k != "token" and k not in self.embedding_dict: continue
                inputs[k].append(v)
            outputs["lm"].append(_outputs["token"])
            outputs["segment"].append(_outputs["segment"])

        if over_length_row_cnt > 0:
            message = "There were total {cnt} over_length_rows.".format(cnt=over_length_row_cnt)
            if self.verbose: print(message)
            logger.info(message)
        return inputs, outputs

    def encode_src(self, src_inputs, src_timesteps: int, src_sep_tokens: List[List[str]], approach: str = "ignore") -> Tuple[Dict[str, List[List[int]]], Dict[str, List[List[int]]], Dict[str, List[List[int]]]]:
        src_inputs_rows = src_inputs.copy()
        src_sep_token_ids = self.get_sep_token_ids(sep_tokens=src_sep_tokens, num_segments=len(src_sep_tokens),
                                                   tokenizer=self.tokenizer)

        inputs = dict()
        inputs["token"] = []
        for k, v in self.embedding_dict.items():
            inputs[k] = []

        over_length_row_cnt = 0
        src_lower_bound = sum([len(v) for v in src_sep_token_ids])
        for row_idx, src_input_row in enumerate(src_inputs_rows):
            status = 0
            _inputs, _ = self.encode_input_row(src_input_row=src_input_row, tgt_input_row=None, src_sep_token_ids=src_sep_token_ids)
            self.assert_isequal_elements_length(data=list(_inputs.values()))

            if self.is_proper_length(ids=_inputs["token"], upper_bound=src_timesteps, lower_bound=src_lower_bound):
                _inputs["token"] = self.pad_row(ids=_inputs["token"], timesteps=src_timesteps,
                                                padding_value=self.src_tokenizer.special_token_dict["pad"]["id"])
                for k, v in _inputs.items():
                    if k == "token": continue
                    v = self.pad_row(ids=v, timesteps=src_timesteps,
                                     padding_value=self.src_tokenizer.special_token_dict["pad"]["id"])
                    _inputs[k] = v
            else:
                if approach == "stop":
                    status = 1
                elif approach == "ignore":
                    status = 1
                elif approach == "truncate":
                    status = 1
                    _inputs["token"] = self.truncate_over_length(ids=_inputs["token"], upper_bound=src_timesteps - 1)
                    _inputs["token"] = self.attach_token(ids=_inputs["token"], append_head=None,
                                                         append_tail=src_sep_token_ids[-1][1])
                    for k, v in _inputs.items():
                        if k == "token": continue
                        v = self.truncate_over_length(ids=v, upper_bound=src_timesteps - 1)
                        v = self.attach_token(ids=v, append_head=None, append_tail=v[-1])
                        _inputs[k] = v

            if status > 0:
                self._raise_approach_error(approach=approach, row_idx=row_idx)
                over_length_row_cnt += 1
                if approach == "ignore": continue

            for k, v in _inputs.items():
                if k != "token" and k not in self.embedding_dict: continue
                inputs[k].append(v)

        if over_length_row_cnt > 0:
            message = "There were total {cnt} over_length_rows.".format(cnt=over_length_row_cnt)
            if self.verbose: print(message)
            logger.info(message)
        return status, inputs

    def sentence_to_ids(self, sentence: str, mask: bool = False) -> List[int]:
        token_ids = HuggingFacePreprocessorInterface.sentence_to_ids(self=self, sentence=sentence, tokenizer=self.tokenizer, mask=mask)
        return token_ids

    def src_sentence_to_ids(self, sentence: str, mask: bool = False) -> List[int]:
        token_ids = HuggingFacePreprocessorInterface.sentence_to_ids(self=self, sentence=sentence, mask=mask, tokenizer=self.src_tokenizer)
        return token_ids

    def tgt_sentence_to_ids(self, sentence: str, mask: bool = False) -> List[int]:
        token_ids = HuggingFacePreprocessorInterface.sentence_to_ids(self=self, sentence=sentence, mask=mask, tokenizer=self.tgt_tokenizer)
        return token_ids

    def get_src_token_length(self, sentence: str) -> int:
        length = HuggingFacePreprocessorInterface.get_token_length(self=self, sentence=sentence, tokenizer=self.src_tokenizer)
        return length

    def get_tgt_token_length(self, sentence: str) -> int:
        length = HuggingFacePreprocessorInterface.get_token_length(self=self, sentence=sentence, tokenizer=self.tgt_tokenizer)
        return length

    def decode(self, rows: List[List[int]], eos_token_id: int = None, keep_pad: bool = False) -> List[str]:
        self.assert_isinstance_list(rows, "rows")
        output = []
        for row in rows:
            if not keep_pad:
                row = [token_id for token_id in row if token_id != self.tokenizer.special_token_dict["pad"]["id"]]
            if eos_token_id is not None:
                eos_token_idx = get_last_index(obj=row, value=eos_token_id)
                row = row[:eos_token_idx + 1]
            row = self.tokenizer.decode(ids=row)
            output.append(row)
        return output

    def src_decode(self, rows: List[List[int]], eos_token_id: int = None, keep_pad: bool = False) -> List[str]:
        self.assert_isinstance_list(rows, "rows")
        output = []
        for row in rows:
            if not keep_pad:
                row = [token_id for token_id in row if token_id != self.src_tokenizer.special_token_dict["pad"]["id"]]
            if eos_token_id is not None:
                eos_token_idx = get_last_index(obj=row, value=eos_token_id)
                row = row[:eos_token_idx + 1]
            row = self.tokenizer.decode(ids=row)
            output.append(row)
        return output

    def tgt_decode(self, rows: List[List[int]], eos_token_id: int = None, keep_pad: bool = False) -> List[str]:
        self.assert_isinstance_list(rows, "rows")
        output = []
        for row in rows:
            if not keep_pad:
                row = [token_id for token_id in row if token_id != self.tgt_tokenizer.special_token_dict["pad"]["id"]]
            if eos_token_id is not None:
                eos_token_idx = get_last_index(obj=row, value=eos_token_id)
                row = row[:eos_token_idx + 1]
            row = self.tokenizer.decode(ids=row)
            output.append(row)
        return output

    def get_token_length(self, sentence: str) -> int:
        length = HuggingFacePreprocessorInterface.get_token_length(self=self, sentence=sentence, tokenizer=self.tokenizer, language=self.language)
        return length

    def save_tokenizer(self, path):
        if not path.endswith("/"): path = path + "/"
        path = path + ModelFilenameConstants.SPM_MODEL_DIR
        self.tokenizer.save_model(path=path)