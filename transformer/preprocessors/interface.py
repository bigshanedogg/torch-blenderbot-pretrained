import re
import json
import logging
import numpy as np
from typing import List, Tuple
from transformer.utils.logger import get_logger
from transformer.utils.common import get_nth_index, get_randint_except
from transformer.utils.tokenizer import MecabTokenizer, SpmTokenizer, HuggingFaceTokenizer
from transformer.assertions.object_assertion import PreprocessorAssertion

logger = get_logger(name=__name__)

class PreprocessorInterface(PreprocessorAssertion):
    mecab_tokenizer = None
    tokenizer = None
    src_tokenizer = None
    tgt_tokenizer = None
    config = None
    verbose = False

    def __init__(self, config_path: str = "./config/preprocessor_config.json", verbose=False) -> None:
        self.mecab_tokenizer = MecabTokenizer()
        with open(config_path, "r", encoding="UTF-8") as fp:
            self.config = json.load(fp)
        self.config["keep_mask_ratio"] = 1.0 - self.config["random_mask_ratio"] - self.config["skip_mask_ratio"]
        self.verbose = verbose

    def preprocess(self, sentence: str) -> str:
        # regex preprocess
        if self.config["remove_symbols"]:
            # remain allowed characters only
            remain_regex = self.config["remain_regex"]
            remain_regex_pattern = re.compile(remain_regex)
            sentence = remain_regex_pattern.sub(" ", sentence)

        if self.config["remove_bracket"]:
            # replace braket to space
            bracket_regex = self.config["bracket_regex"]
            bracket_regex_pattern = re.compile(bracket_regex)
            sentence = bracket_regex_pattern.sub(" ", sentence)

        if self.config["replace_whitespace"]:
            # replace whitespace to a single space
            whitespace_regex_pattern = re.compile(self.config["whitespace_regex"])
            sentence = whitespace_regex_pattern.sub(" ", sentence)
        return sentence

    def sentence_to_ids(self, sentence: str, mask: bool = False, tokenizer: SpmTokenizer = None, language: str = None) -> List[int]:
        if not isinstance(sentence, str): sentence = str(sentence)
        sentence = self.preprocess(sentence=sentence)
        tokens = None
        if language == "eng": tokens = self.mecab_tokenizer.tokenize_eng(sentence=sentence, return_pos=False)
        elif language == "kor": tokens = self.mecab_tokenizer.tokenize_kor(sentence=sentence, return_pos=False)
        token_ids = tokenizer.tokens_to_ids(tokens=tokens, mask=mask)
        return token_ids

    def get_token_length(self, sentence: str, tokenizer: SpmTokenizer, language: str) -> int:
        tokens = None
        if language == "eng":
            tokens = self.mecab_tokenizer.tokenize_eng(sentence=sentence, return_pos=False)
        elif language == "kor":
            tokens = self.mecab_tokenizer.tokenize_kor(sentence=sentence, return_pos=False)
        tokens = tokenizer.tokens_to_ids(tokens=tokens, mask=False)
        length = len(tokens)
        return length

    def is_proper_length(self, ids: List[int], upper_bound: int, lower_bound: int = 0) -> bool:
        if len(ids) > lower_bound and len(ids) <= upper_bound: return True
        else: return False

    def truncate_over_length(self, ids: List[int], upper_bound: int) -> List[int]:
        ids = ids[:upper_bound]
        return ids

    def pad_row(self, ids: List[int], timesteps: int, padding_value: int, post: bool = True) -> List[int]:
        self.assert_isinstance_list(data=ids, parameter_name="ids")
        output = None
        padding_len = (timesteps - len(ids))
        padding_vector = [padding_value] * padding_len
        if post: output = ids + padding_vector
        else: output = padding_vector + ids
        return output

    def onehot_ids(self, ids: List[int], num_class: int) -> List[List[float]]:
        self.assert_isinstance_list(data=ids, parameter_name="ids")
        output = np.zeros((len(ids), num_class))
        output = output.tolist()
        for idx, _id in enumerate(ids):
            output[idx][_id] = 1.0
        return output

    def get_sep_token_ids(self, sep_tokens, num_segments, tokenizer: SpmTokenizer):
        if sep_tokens is None:
            # sep_tokens = [["cls", "sep"], [None, "sep"], [None, "sep"]]
            # e.g.) <cls> ~ <sep> ~ <sep> ~ <sep>
            sep_token_ids = [[None, tokenizer.special_token_dict["sep"]["id"]]] * (num_segments - 1)
            sep_token_ids.insert(0, [tokenizer.special_token_dict["cls"]["id"], tokenizer.special_token_dict["sep"]["id"]])
            return sep_token_ids
        else:
            # sep_tokens = [["persona", None], ["context", None], ["candidate", None]]
            # e.g.) <pers> ~ <ctxt> ~ <cand> ~
            sep_token_ids = []
            for special_token_1, special_token_2 in sep_tokens:
                special_token_1_id = None
                if special_token_1 is not None:
                    self.assert_isin_obj(element=special_token_1, obj=tokenizer.special_token_dict.keys())
                    special_token_1_id = tokenizer.special_token_dict[special_token_1]["id"]
                special_token_2_id = None
                if special_token_2 is not None:
                    self.assert_isin_obj(element=special_token_2, obj=tokenizer.special_token_dict.keys())
                    special_token_2_id = tokenizer.special_token_dict[special_token_2]["id"]
                sep_token_ids.append([special_token_1_id, special_token_2_id])
            return sep_token_ids

    def attach_token(self, ids: List[int], append_head: int = None, append_tail: int = None) -> List[int]:
        self.assert_isinstance_list(ids, parameter_name="ids")
        output = ids.copy()
        if append_head is not None: output = [append_head] + output
        if append_tail is not None: output = output + [append_tail]
        return output

    def get_negative_sample_pairs(self, inputs_size: int):
        negative_sample_pairs = []
        for src_idx in range(0, inputs_size):
            random_idx = get_randint_except(low=0, high=inputs_size, except_value=src_idx)
            negative_sample_pair = (src_idx, random_idx)
            negative_sample_pairs.append(negative_sample_pair)
        return negative_sample_pairs

    def get_common_subtokens(self, sentence_a: str, sentence_b, tokenizer: SpmTokenizer = None, language: str = None):
        subtokens_a = self.sentence_to_subtokens(sentence=sentence_a, tokenizer=tokenizer, language=language)
        subtokens_b = self.sentence_to_subtokens(sentence=sentence_b, tokenizer=tokenizer, language=language)
        common_subtokens = set(subtokens_a).intersection(set(subtokens_b))
        common_subtokens = list(common_subtokens)
        return common_subtokens

    def sentence_to_subtokens(self, sentence: str, tokenizer: SpmTokenizer, language: str = None):
        tokens = None
        if language == "eng":
            tokens = self.mecab_tokenizer.tokenize_eng(sentence=sentence, return_pos=False)
        elif language == "kor":
            tokens = self.mecab_tokenizer.tokenize_kor(sentence=sentence, return_pos=False)
        subtokens = tokenizer.tokens_to_pieces(tokens=tokens, mask=False)
        return subtokens

    def get_common_tokens(self, sentence_a: str, sentence_b, language: str = None):
        tokens_a = self.sentence_to_tokens(sentence=sentence_a, language=language)
        tokens_b = self.sentence_to_tokens(sentence=sentence_a, language=language)
        common_tokens = set(tokens_a).intersection(set(tokens_b))
        common_tokens = list(common_tokens)
        return common_tokens

    def sentence_to_tokens(self, sentence: str, language: str = None):
        tokens = None
        if language == "eng":
            tokens = self.mecab_tokenizer.tokenize_eng(sentence=sentence, return_pos=False)
        elif language == "kor":
            tokens = self.mecab_tokenizer.tokenize_kor(sentence=sentence, return_pos=False)
        return tokens

    def save_tokenizer(self, path):
        self.assert_implemented(method_name="save_spm_tokenizer")

    def _raise_approach_error(self, approach: str, row_idx: int = None) -> None:
        if approach == "stop":
            error_message_template = "There is an over_length_row - sentence_idx"
            if row_idx is not None: error_message_template += ":{row_idx}".format(row_idx=row_idx)
            message = error_message_template.format(sentence_idx=row_idx)
            if self.verbose: print(message)
            logging.error(message)
            logger.error(message)
            raise AssertionError(message)
        elif approach == "ignore":
            error_message_template = "{approach} over_length_row - sentence_idx"
            if row_idx is not None: error_message_template += ":{row_idx}".format(row_idx=row_idx)
            message = error_message_template.format(approach=approach)
            logger.debug(message)
            return None
        elif approach == "truncate":
            error_message_template = "{approach} over_length_row - sentence_idx"
            if row_idx is not None: error_message_template += ":{row_idx}".format(row_idx=row_idx)
            message = error_message_template.format(approach=approach)
            logger.debug(message)
            return None

class HuggingFacePreprocessorInterface(PreprocessorInterface):
    mecab_tokenizer = None
    tokenizer = None
    src_tokenizer = None
    tgt_tokenizer = None
    config = None
    verbose = False

    def __init__(self, config_path: str = "./config/preprocessor_config.json", verbose=False):
        PreprocessorInterface.__init__(self, config_path=config_path, verbose=verbose)
        self.tokenizer = HuggingFaceTokenizer()

    def sentence_to_ids(self, sentence: str, tokenizer: HuggingFaceTokenizer, mask: bool, clean: bool = False):
        if clean: sentence = self.preprocess(sentence)
        token_ids = tokenizer.sentence_to_ids(sentence=sentence)
        return token_ids

    def get_token_length(self, sentence: str, tokenizer: HuggingFaceTokenizer, language: str) -> int:
        token_ids = self.setnence_to_ids(sentence=sentence)
        length = len(token_ids)
        return length

    def sentence_to_subtokens(self, sentence: str, tokenizer: HuggingFaceTokenizer, language: str = "kor"):
        subtokens = self.tokenizer.sentence_to_pieces(sentence=sentence)
        return subtokens