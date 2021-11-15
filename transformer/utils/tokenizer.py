import os
import json
import nltk
import pickle
import shutil
import logging
import sentencepiece as spm
import numpy as np
from datetime import datetime
from transformers import PreTrainedTokenizerFast
from typing import List, Tuple
from transformer.assertions.object_assertion import TokenizerAssertion

MecabConfig = {
    "nltk_prefix": "nltk_{tag}",
    "mecab_prefix": "mecab_{tag}",
    "mecab_eng_tag": "SL",
    "mecab_tag_delimiter": "+",
    "interjection_normalize_pos": ["IC"],
    "arabia_normalize_pos": ["NR", "MM"],
    "chinese_normalize_pos": ["NNG", "XPN", "XR", "XSN"],
    "keyword_pos": ["NNP", "NNBC", "NNG", "NNB" , "VCN", "VA", "VV", "VX", "VCP", "SL"],
    "a2h":{
        "1": ["하나", "한", "일", "一"],
        "2": ["둘", "두", "이", "二"],
        "3": ["셋", "세", "서", "삼", "三"],
        "4": ["넷", "네", "너", "사", "四"],
        "5": ["다섯", "다서", "대", "댓", "오", "五"],
        "6": ["여섯", "여서", "예", "육", "륙", "六"],
        "7": ["일곱", "일고", "닐곱", "칠", "七"],
        "8": ["여덟", "여더", "여덜", "팔", "八"],
        "9": ["아홉", "구", "九"],
        "10": ["열", "십", "十"],
        "20": ["스물", "이십"],
        "30": ["서른", "삼십"],
        "40": ["마흔", "사십"],
        "50": ["쉰", "오십"],
        "60": ["예순", "육십"],
        "70": ["일흔", "칠십"],
        "80": ["여든", "팔십"],
        "90": ["아흔", "구십"],
        "100": ["백", "百"],
        "1000": ["천", "千"],
        "10000": ["만", "萬"],
        "100000000": ["억", "億"],
        "1000000000000": ["조", "兆"]
    },
    "decimal_arabia": [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000]
}

class MecabTokenizer(TokenizerAssertion):
    tagger = None
    normalize_interjection = None
    normalize_arabia = None
    normalize_chinese = None

    def __init__(self, normalize_interjection=True, normalize_arabia=False, normalize_chinese=False):
        self.normalize_interjection = normalize_interjection
        self.normalize_arabia = normalize_arabia
        self.normalize_chinese = normalize_chinese
        try:
            from konlpy.tag import Mecab
            self.tagger = Mecab()
            message = "Imported konlpy.tag.Mecab successfully"
            print(message)
            logging.info(message)
        except  Exception as ex:
            error_message = "Cannot Import konlpy Mecab tagger: {error_type} - {error}".format(error_type=type(ex), error=ex)
            print(error_message)
            message = "Importing MeCab for Windows"
            print(message)
            try:
                import MeCab
                self.tagger = MeCab.Tagger()
            except:
                import mecab
                self.tagger = mecab.Tagger()
            self.tagger.pos = self._parse_simplify
            message = "Imported MeCab for Windows successfully"
            print(message)
            logging.info(message)
        self._load_config(config=MecabConfig)

    def _load_config(self, config: dict):
        json_data = config
        self.nltk_prefix = json_data["nltk_prefix"]
        self.mecab_prefix = json_data["mecab_prefix"]
        self.mecab_eng_tag = json_data["mecab_eng_tag"]
        self.mecab_tag_delimiter = json_data["mecab_tag_delimiter"]
        self.interjection_normalize_pos = json_data["interjection_normalize_pos"]
        self.arabia_normalize_pos = json_data["arabia_normalize_pos"]
        self.chinese_normalize_pos = json_data["chinese_normalize_pos"]
        self.keyword_pos = json_data["keyword_pos"]
        self.a2h = {int(a):h for a,h in json_data["a2h"].items()}
        self.h2a = {h:int(a) for a,h_list in json_data["a2h"].items() for h in h_list}
        self.decimal_arabia = json_data["decimal_arabia"]

    def _parse_simplify(self, sentence):
        '''
        윈도우 MeCab의 parse 메소드를 konly.tag.Meacb의 pos로 변
        불필요한 parsing 정보 제외하고 필요한 정보만 추출
        e.g) '회사\tNNG,*,F,회사,*,*,*,*' => ('회사', 'NNG')
        :param token:
        :param normalize_chinese:
        :return:
        '''
        output = []
        parsed_result = self.tagger.parse(sentence).split("\n")[:-2]
        for token_row in parsed_result:
            word, _pos = token_row.split("\t")[0:2]
            pos_splited = _pos.split(",")
            pos = pos_splited[0]

            if self.normalize_chinese and pos_splited[0] in self.chinese_normalize_pos:
                # 한자어일 경우, 한글 득음으로 정규화
                word = pos_splited[3]
            row = (word, pos)
            output.append(row)
        return output

    def tokenize_kor(self, sentence: str, return_pos: bool = False) -> List:
        output = []
        sentence_token = self.pos(sentence)
        for token, tag in sentence_token:
            if return_pos:
                tag = self._convert_mecab_tag_to_pos(tag)
                if tag == self.mecab_eng_tag:
                    _, tag = nltk.pos_tag([token])[0]
                    output.append((token, self.nltk_prefix.format(tag=tag)))
                else:
                    output.append((token, self.mecab_prefix.format(tag=tag)))
            else:
                output.append(token)
        return output

    def tokenize_eng(self, sentence: str, return_pos: bool = False) -> List:
        output = []
        sentence_token = nltk.pos_tag(sentence.split())
        for token, tag in sentence_token:
            if return_pos:
                output.append((token, self.nltk_prefix.format(tag=tag)))
            else:
                output.append(token)
        return output

    def _convert_mecab_tag_to_pos(self, tag: str) -> str:
        pos_list = tag.split(self.mecab_tag_delimiter)
        output = pos_list[0]
        return output

    def pos(self, sentence: str) -> List[Tuple[str, str]]:
        output = self.tagger.pos(sentence)
        # interjection_normalize
        if self.normalize_interjection: output = self._normalize_interjection(parsed_tokens=output)
        # arabia_normalize
        if self.normalize_arabia: output = self._normalize_arabia(parsed_tokens=output)
        return output

    def extract_keywords(self, sentence: str) -> List[str]:
        parsed_tokens = self.tagger.pos(sentence)
        keywords = [word for word,pos in parsed_tokens if pos in self.keyword_pos]
        return keywords

    def _normalize_interjection(self, parsed_tokens):
        idx = 0
        while True:
            if idx >= len(parsed_tokens) - 1: break
            cur_token_part = parsed_tokens.pop(idx)
            next_token_part = parsed_tokens.pop(idx)
            if cur_token_part[1] == next_token_part[1] and cur_token_part[1] in self.interjection_normalize_pos:
                if set(cur_token_part[0]) <= set(next_token_part[0]):
                    # "아아아아아" 와 "아아아" 인 경우
                    parsed_tokens.insert(idx, cur_token_part)
                else:
                    parsed_tokens.insert(idx, next_token_part)
                continue
            else:
                # "하하하하" 와 "하하하하핫" 인 경우
                if set(cur_token_part[0]).intersection(set(next_token_part[0])) == set(next_token_part[0]):
                    parsed_tokens.insert(idx, cur_token_part)
                    continue
                if set(cur_token_part[0]).intersection(set(next_token_part[0])) == set(cur_token_part[0]):
                    parsed_tokens.insert(idx, next_token_part)
                    continue

            parsed_tokens.insert(idx, next_token_part)
            parsed_tokens.insert(idx, cur_token_part)
            idx += 1
        return parsed_tokens

    def _normalize_arabia(self, parsed_tokens):
        '''
        서수, 기수, 한자형 모두 아라비아 숫자로 정규화
        :param parsed_tokens:
        :return:
        '''
        _result = []
        for idx, (word,pos) in enumerate(parsed_tokens):
            if pos in self.arabia_normalize_pos:
                _, arabia_eujeol, hangul_eujeol = self._split_arabia(word, [], [])
                word_list = []
                for eujeol in arabia_eujeol:
                    idx = word.index(eujeol)
                    word_list.append((idx, (eujeol, "NR")))
                for eujeol in hangul_eujeol:
                    idx = word.index(eujeol)
                    word_list.append((idx, (eujeol, "MM")))

                word_list.sort(key=lambda x: x[0])
                for element in word_list:
                    _result.append(element[1])
            else:
                _result.append((word, pos))
        parsed_tokens = _result

        arabia = []
        _result = []
        for idx, (word, pos) in enumerate(parsed_tokens):
            if pos not in self.arabia_normalize_pos or word not in self.h2a.keys():
                if len(arabia) > 0:
                    _result.append((str(sum(arabia)), "SN")) # "SN": 숫자
                    arabia = []
                row = (word, pos)
                _result.append(row)
                continue

            arabia_token = self.h2a[word]
            if len(arabia) > 0:
                if arabia_token in self.decimal_arabia:
                    prev_arabia_token = arabia.pop()
                    arabia.append(prev_arabia_token * arabia_token)
                    arabia = [sum(arabia)]
                else:
                    if arabia[-1] < 10:
                        _result.append((str(arabia[0]), "SN")) # "SN": 숫자
                        arabia = []
                    arabia.append(arabia_token)
            else:
                arabia.append(arabia_token)

            if len(arabia) > 0 and (idx == len(parsed_tokens) - 1):
                _result.append((str(sum(arabia)), "SN")) # "SN": 숫자
                arabia = []
        return _result

    def _split_arabia(self, token, split_list, unsplit_list):
        '''
        2개 이상의 숫자가 묶여있는 문자열을 나누는 메소드
        :param token:
        :param split_list:
        :param unsplit_list:
        :return:
        '''
        split_list_init, unsplit_list_init = split_list.copy(), unsplit_list.copy()
        for window_size in range(1, len(token)+1):
            split_list, unsplit_list = split_list_init.copy(), unsplit_list_init.copy()
            left = token[:window_size]
            right = token[window_size:]
            if left in self.h2a: split_list.append(left)
            else: unsplit_list.append(left)

            if len(right) < 1:
                return right, split_list, unsplit_list
            else:
                right, split_list, unsplit_list = self._split_arabia(right, split_list, unsplit_list)
                if len(split_list) > len(split_list_init): break
        return right, split_list, unsplit_list


class SpmTokenizer(TokenizerAssertion):
    mlm_ratio = None
    random_mask_ratio = None
    skip_mask_ratio = None
    keep_mask_ratio = None
    special_token_dict = None
    special_token_ids = None
    tokenizer_path = None
    tokenizer = None
    tokens = None
    vocab_size = None
    spm_model_type = None
    spm_model_prefix = "sentence_piece"
    default_spm_input_path = "./spm_input_file.txt"

    def __init__(self, mlm_ratio: float = 0.15, random_mask_ratio: float = 0.1, skip_mask_ratio: float = 0.1, config_path: str = "./config/special_token_dict.json") -> None:
        with open(config_path, encoding="UTF-8") as fp:
            self.special_token_dict = json.load(fp)
        self.special_token_ids = [v["id"] for k, v in self.special_token_dict.items()]
        self.mlm_ratio = mlm_ratio
        self.random_mask_ratio = random_mask_ratio
        self.skip_mask_ratio = skip_mask_ratio
        self.keep_mask_ratio = 1.0 - (random_mask_ratio + skip_mask_ratio)

    def tokens_to_ids(self, tokens: List[str], mask: bool) -> List[int]:
        self.assert_isinstance_list(tokens, "tokens")
        output = []
        for token in tokens:
            token_ids = self.tokenizer.EncodeAsIds(token)
            if mask:
                for token_id in token_ids:
                    random_prob = np.random.rand()
                    if random_prob < self.mlm_ratio:
                        token_id = self._mask_token(token_id=token_id)
                    output.append(token_id)
            else:
                output += token_ids
        return output

    def _mask_token(self, token_id: int) -> int:
        random_prob = np.random.rand()
        if random_prob < self.keep_mask_ratio:
            # keep mask
            token_id = self.special_token_dict["mask"]["id"]
        elif random_prob < self.keep_mask_ratio + self.random_mask_ratio:
            # random mask
            token_id = self.special_token_dict["pad"]["id"]
            while token_id in self.special_token_ids:
                token_id = np.random.choice(list(range(0, len(self.tokens))))
                token_id = int(token_id)
        else:
            # skip mask
            pass
        return token_id

    def tokens_to_pieces(self, tokens: List[str], mask: bool) -> List[str]:
        token_ids = self.tokens_to_ids(tokens=tokens, mask=mask)
        output = [self.tokenizer.IdToPiece(token_id) for token_id in token_ids]
        return output

    def decode(self, ids: List[int]) -> str:
        self.assert_isinstance_list(ids, "ids")
        output = self.tokenizer.Decode(ids)
        return output

    def train_spm_model(self, sentences: List[str], vocab_size: int, path: str = None, spm_model_type: str = "bpe") -> None:
        self.assert_isinstance_list(sentences, "sentences")
        with open(self.default_spm_input_path, "w", encoding="UTF-8") as fp:
            for sentence in sentences:
                fp.write(str(sentence) + "\n")

        self._train_spm_model(input_data_path=self.default_spm_input_path, vocab_size=vocab_size, path=path, spm_model_type=spm_model_type)
        os.remove(self.default_spm_input_path)

        message = "trained spm_model with {size} sentences".format(size=len(sentences))
        print(message)
        logging.info(message)

    def _train_spm_model(self, input_data_path: str, vocab_size: int, path: str = None, spm_model_type: str = "bpe") -> None:
        today_date = datetime.today().strftime("%Y%m%d")[2:]
        spm_model_path = "./spm_v{vocab_size}_{today}/".format(vocab_size=vocab_size, today=today_date)
        if path is None: print("default spm_model_path: '{path}'".format(path=spm_model_path))
        if not os.path.exists(spm_model_path): os.makedirs(spm_model_path)

        spm_cmd = self._get_spm_cmd(input_data_path=input_data_path, spm_model_path=spm_model_path, spm_model_type=spm_model_type, vocab_size=vocab_size)

        # train
        spm.SentencePieceTrainer.Train(spm_cmd)
        with open(spm_model_path + "special_token_dict.pickle", "wb") as fp:
            pickle.dump(self.special_token_dict, fp)

        # load trained models
        spm_model_file = spm_model_path + self.spm_model_prefix + ".model"
        spm_vocab_file = spm_model_path + self.spm_model_prefix + ".vocab"
        spm_model = spm.SentencePieceProcessor()
        spm_model.Load(spm_model_file)

        spm_tokens = []
        with open(spm_vocab_file, "r", encoding="UTF-8") as fp:
            for row in fp:
                token, idx = row.strip().split("\t")
                spm_tokens.append(token)

        self.tokenizer = spm_model
        self.tokens = spm_tokens
        self.vocab_size = self.tokenizer.GetPieceSize()
        self.tokenizer_path = spm_model_path
        self.spm_model_type = spm_model_type

        if path is not None:
            self.save_model(path=path, copy=False)
            self.tokenizer_path = path

    def _get_spm_cmd(self, input_data_path: str, spm_model_path: str, spm_model_type: str, vocab_size: int) -> str:
        special_token_dict = self.special_token_dict.copy()

        # setting training command
        # e.g.) '--input=./spm_input_file.txt --model_prefix=test --vocab_size=10000 --model_type=bpe'
        spm_cmd_template = "--input={input_path} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type}"
        spm_cmd = spm_cmd_template.format(input_path=input_data_path, model_prefix=spm_model_path + self.spm_model_prefix, vocab_size=vocab_size, model_type=spm_model_type)

        # e.g.) '--bos_id=1 --bos_piece=<s> --unkid=3 --unk_piece=<unk> --eos_id=2 --eos_piece=</s> --pad_id=0 --pad_piece=<pad>'
        pad = special_token_dict.pop("pad")
        bos = special_token_dict.pop("bos")
        eos = special_token_dict.pop("eos")
        unk = special_token_dict.pop("unk")
        token_append_template = " --{token_type}_id={token_id} --{token_type}_piece={token}"
        token_append_cmd = \
            token_append_template.format(token_type="pad", token_id=pad["id"], token=pad["token"]) + \
            token_append_template.format(token_type="bos", token_id=bos["id"], token=bos["token"]) + \
            token_append_template.format(token_type="eos", token_id=eos["id"], token=eos["token"]) + \
            token_append_template.format(token_type="unk", token_id=unk["id"], token=unk["token"])

        # e.g.) ' --user_defined_symbols=<cls>,<sep>,<turn>,<mask>,<num>'
        user_defined_symbols = [(v["token"], v["id"]) for k,v in special_token_dict.items()]
        sorted(user_defined_symbols, key=lambda x:x[1])
        user_defined_symbols = [token for token,id in user_defined_symbols]
        uds_append_template = " --user_defined_symbols={user_defined_symbols}"
        user_defined_symbols = ",".join(user_defined_symbols)
        uds_cmd = uds_append_template.format(user_defined_symbols=user_defined_symbols)
        spm_cmd = spm_cmd + token_append_cmd + uds_cmd
        return spm_cmd

    def load_model(self, path: str) -> None:
        if not path.endswith("/"): path = path + "/"
        self.tokenizer_path = path
        spm_model_file = self.tokenizer_path + self.spm_model_prefix + ".model"
        spm_vocab_file = self.tokenizer_path + self.spm_model_prefix + ".vocab"
        special_token_dict_file = self.tokenizer_path + "special_token_dict.pickle"

        # spm_model_file
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(spm_model_file)
        self.vocab_size = self.tokenizer.GetPieceSize()


        # spm_vocab_file
        spm_tokens = []
        with open(spm_vocab_file, "r", encoding="UTF-8") as fp:
            for row in fp:
                token, idx = row.strip().split("\t")
                spm_tokens.append(token)
        self.tokens = spm_tokens

        # special_token_dict_file
        special_token_dict = None
        with open(special_token_dict_file, "rb") as fp:
            special_token_dict = pickle.load(fp)
        self.special_token_dict = special_token_dict

        message = "loaded spm_model: '{path}'".format(path=path)
        print(message)
        logging.info(message)

    def save_model(self, path: str, copy: bool = False) -> None:
        if path != self.tokenizer_path:
            if not path.endswith("/"): path = path + "/"
            if os.path.isdir(path): shutil.rmtree(path)
            shutil.copytree(self.tokenizer_path, path)

        message = "copied spm_model: '{path}'".format(path=path)
        if not copy:
            shutil.rmtree(self.tokenizer_path)
            message = "saved tokenizer: '{path}'".format(path=path)

        print(message)
        logging.info(message)
        if not copy:
            self.tokenizer_path = path

class HuggingFaceTokenizer(TokenizerAssertion):
    special_token_dict = None
    tokenizer = None
    tokens = None
    vocab_size = None
    model_path = None

    def __init__(self, config_path: str = "./config/special_token_dict.json"):
        with open(config_path, encoding="UTF-8") as fp:
            self.special_token_dict = json.load(fp)

    def load_pretrained_model(self, name_or_path: str, add_special_token:bool = True) -> None:
        special_token_dict = self.special_token_dict.copy()
        base_special_tokens = ["pad", "bos", "eos", "unk", "mask", "cls", "sep"]
        special_token_params = dict()
        for base_special_token in base_special_tokens:
            k = base_special_token + "_token"
            v = special_token_dict.pop(base_special_token)["token"]
            special_token_params[k] = v

        additional_special_tokens_names = []
        additional_special_tokens = []
        if add_special_token:
            additional_special_tokens_names = []
            additional_special_tokens = []
            for special_token_name, special_token in special_token_dict.items():
                additional_special_tokens_names.append(special_token_name)
                additional_special_tokens.append(special_token["token"])

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=name_or_path, additional_special_tokens=additional_special_tokens, **special_token_params)
        self.update_info()

        message = "loaded pretrained huggingface_tokenizer: '{name_or_path}'".format(name_or_path=name_or_path)
        print(message)
        logging.info(message)

    def update_info(self):
        special_token_dict = dict()
        for special_token_id, special_token in zip(self.tokenizer.all_special_ids, self.tokenizer.all_special_tokens):
            for k, v in self.special_token_dict.items():
                if v["token"] == special_token:
                    v["id"] = special_token_id
                    special_token_dict[k] = v
        self.special_token_dict = special_token_dict
        self.tokens = self.tokenizer.get_vocab()
        self.vocab_size = len(self.tokenizer)

    def sentence_to_ids(self, sentence:str) -> List[int]:
        self.assert_isintance(obj=sentence, data_type=str)
        output = self.tokenizer.encode(sentence)
        return output

    def sentence_to_pieces(self, sentence: str) -> List[str]:
        output = self.tokenizer.tokenize(sentence=sentence)
        return output

    def decode(self, ids: List[int]) -> str:
        self.assert_isinstance_list(ids, "ids")
        output = self.tokenizer.decode(ids)
        return output

    def load_model(self, path):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=path)
        self.tokenizer = tokenizer
        self.update_info()
        self.model_path = path
        message = "loaded huggingface_tokenizer: '{path}'".format(path=path)
        print(message)

    def save_model(self, path):
        self.tokenizer.save_pretrained(path)
        self.model_path = path

        message = "saved huggingface_tokenizer: '{path}'".format(path=path)
        print(message)
        logging.info(message)