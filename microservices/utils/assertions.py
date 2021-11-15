import os
from transformer.preprocessors.sentence_bert_preprocessor import SentenceBertPreprocessor

def assert_path_exists(path):
    assert_message = "Not exist path: {path}".format(path=path)
    if not os.path.exists(path):
        raise AssertionError(assert_message)

def assert_is_dir(path):
    assert_message = "Not a Directory: {path}".format(path=path)
    if not os.path.isdir(path):
        raise AssertionError(assert_message)

def assert_isloaded_model(model):
    assert_message = "Model has not been loaded"
    if model is None:
        raise AssertionError(assert_message)

def assert_isloaded_preprocessor(preprocessor):
    assert_message = "SentenceBertPreprocessor has not been loaded"
    if preprocessor is None or not isinstance(preprocessor, SentenceBertPreprocessor):
        raise AssertionError(assert_message)

def assert_isloaded_candidates(candidates):
    assert_message = "candidates has not been loaded"
    if candidates is None:
        raise AssertionError(assert_message)