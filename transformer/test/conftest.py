import pytest
import os
print("dir:", os.getcwd())

from transformer.utils.tokenizer import MecabTokenizer
# from transformer.preprocessor.interface import Preprocessor


@pytest.fixture
def mecab_tokenizer():
    normalize_interjection = True
    normalize_arabia = False
    normalize_chinese = False
    mecab = MecabTokenizer(normalize_interjection=normalize_interjection, normalize_arabia=normalize_arabia, normalize_chinese=normalize_chinese)
    return mecab

@pytest.fixture
def mecab_tokenizer_normalize_arabia():
    normalize_interjection = False
    normalize_arabia = True
    normalize_chinese = False
    mecab = MecabTokenizer(normalize_interjection=normalize_interjection, normalize_arabia=normalize_arabia, normalize_chinese=normalize_chinese)
    return mecab

@pytest.fixture
def mecab_tokenizer_normalize_chinese():
    normalize_interjection = False
    normalize_arabia = False
    normalize_chinese = True
    mecab = MecabTokenizer(normalize_interjection=normalize_interjection, normalize_arabia=normalize_arabia, normalize_chinese=normalize_chinese)
    return mecab

# @pytest.fixture
# def preprocessor_fixture():
#     timeseteps = 128
#     spm_model_path = "../data/kor_wiki_spm_model_v5000"
#     prep = Preprocessor(timeseteps=timeseteps)
#     prep.load_spm_model(path=spm_model_path)
#     return prep