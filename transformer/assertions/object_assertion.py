import torch
import logging
from transformer.assertions.interface import Assertion, AssertConditionConstants

class TokenizerAssertion(Assertion):
    def assert_isloaded_tokenizer(self):
        assert_message = "SentencePiece model (spm_model) has not been loaded"
        if self.tokenizer is None:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isloaded_nltk_wordset(self):
        assert_message = "Check if following command works:\n>> import nltk\n>> nltk.download('wordnet')"
        if len(self.nltk_wordest) <= 0:
            logging.error(assert_message)
            raise AssertionError(assert_message)

class PreprocessorAssertion(Assertion):
    def assert_isin_approaches(self, approach):
        approaches = AssertConditionConstants.preprocess_approaches
        assert_message = "approach must be in {approaches}".format(approaches=approaches)
        if approach not in approaches:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isin_languages(self, language):
        languages = AssertConditionConstants.languages
        assert_message = "approach must be in {languages}".format(languages=languages)
        if language not in languages:
            logging.error(assert_message)
            raise AssertionError(assert_message)

class DataAssertion(Assertion):
    def assert_isinstance_preprocessor(self, preprocessor):
        assert_message = "The data type of parameter 'preprocessor' must be transformer.utils.preprocessor.Preprocessor"
        if not isinstance(preprocessor, PreprocessorAssertion):
            logging.error(assert_message)
            raise AssertionError(assert_message)

class ModelAssertion(Assertion):
    @classmethod
    def assert_implemented_(cls, method_name):
        assert_message = "{method_name} method must be implemented".format(method_name=method_name)
        logging.error(assert_message)
        raise AssertionError(assert_message)

    def assert_isinstance_loss(self, loss_function, loss_name):
        assert_message = "The data type of parameter '{loss_name}' must be torch.nn.modules.loss._Loss".format(loss_name=loss_name)
        if not isinstance(loss_function, torch.nn.modules.loss._Loss):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isinstance_optimizer(self, optimizer):
        assert_message = "The data type of parameter 'optimizer' must be torch.optim.Optimizer"
        if not isinstance(optimizer, torch.optim.Optimizer):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isinstance_model(self, model):
        assert_message = "The data type of parameter 'models' must be torch.nn.modules.Module"
        if not isinstance(model, torch.nn.modules.Module):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isin_models(self, model):
        models = AssertConditionConstants.available_models
        assert_message = "models must be in {models}".format(models=models)
        if model not in models:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isin_aggregation_methods(self, method):
        methods = AssertConditionConstants.aggregation_methods
        assert_message = "method must be in {models}".format(models=methods)
        if method not in methods:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isin_optimizers(self, optimizer):
        optimizers = AssertConditionConstants.available_optimizers
        assert_message = "optimizers must be in {optimizers}".format(optimizers=optimizers)
        if optimizer not in optimizers:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isin_decoding_methods(self, method):
        methods = AssertConditionConstants.decoding_methods
        assert_message = "method must be in {methods}".format(methods=methods)
        if method not in methods:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isin_metrics(self, metric):
        metrics = AssertConditionConstants.metrics
        assert_message = "metric must be in {models}".format(models=metrics)
        if metric not in metrics:
            logging.error(assert_message)
            raise AssertionError(assert_message)

class TrainerAssertion(Assertion):
    @classmethod
    def assert_implemented_(cls, method_name):
        assert_message = "{method_name} method must be implemented".format(method_name=method_name)
        logging.error(assert_message)
        raise AssertionError(assert_message)

    @classmethod
    def assert_contain_elements_(self, required, target, name=None):
        assert_message = "Data must contain following element: {element}"
        if name is not None: assert_message = "{name} must contain following element: '{{element}}'".format(name=name)
        for element in required:
            if element not in target:
                assert_message = assert_message.format(element=element)
                logging.error(assert_message)
                raise AssertionError(assert_message)

    @classmethod
    def assert_isinstance_preprocessor_(cls, preprocessor):
        assert_message = "The data type of parameter 'preprocessor' must be transformer.utils.preprocessor.Preprocessor"
        if not isinstance(preprocessor, PreprocessorAssertion):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isinstance_preprocessor(self, preprocessor):
        assert_message = "The data type of parameter 'preprocessor' must be transformer.utils.preprocessor.Preprocessor"
        if not isinstance(preprocessor, PreprocessorAssertion):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    @classmethod
    def assert_isinstance_optimizer_(cls, optimizer):
        assert_message = "The data type of parameter 'optimizer' must be torch.optim.Optimizer"
        if not isinstance(optimizer, torch.optim.Optimizer):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isinstance_optimizer(self, optimizer):
        assert_message = "The data type of parameter 'optimizer' must be torch.optim.Optimizer"
        if not isinstance(optimizer, torch.optim.Optimizer):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    @classmethod
    def assert_isinstance_model_(cls, model):
        assert_message = "The data type of parameter 'models' must be torch.nn.modules.Module"
        if not isinstance(model, torch.nn.modules.Module):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isinstance_model(self, model):
        assert_message = "The data type of parameter 'models' must be torch.nn.modules.Module"
        if not isinstance(model, torch.nn.modules.Module):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isin_models(self, model):
        models = AssertConditionConstants.available_models
        assert_message = "models must be in {models}".format(models=models)
        if model not in models:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isin_optimizers(self, optimizer):
        optimizers = AssertConditionConstants.available_optimizers
        assert_message = "optimizers must be in {optimizers}".format(optimizers=optimizers)
        if optimizer not in optimizers:
            logging.error(assert_message)
            raise AssertionError(assert_message)

class ServiceAssertion(Assertion):
    def assert_isloaded_model(self):
        assert_message = "Model has not been loaded"
        if self.model is None:
            raise AssertionError(assert_message)

    def assert_isloaded_preprocessor(self):
        assert_message = "Preprocessor has not been loaded"
        if self.preprocessor is None or not isinstance(self.preprocessor, PreprocessorAssertion):
            raise AssertionError(assert_message)

    def assert_isloaded_candidates(self):
        assert_message = "Candidates has not been loaded"
        if self.candidates is None:
            raise AssertionError(assert_message)

    def assert_isloaded_tokenizer(self):
        assert_message = "Tokenizer has not been loaded"
        if self.tokenizer is None:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isloaded_src_tokenizer(self):
        assert_message = "SrcTokenizer has not been loaded"
        if self.src_tokenizer is None:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isloaded_tgt_tokenizer(self):
        assert_message = "TgtTokenizer has not been loaded"
        if self.tgt_tokenizer is None:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isin_decoding_methods(self, method):
        methods = AssertConditionConstants.decoding_methods
        assert_message = "method must be in {methods}".format(methods=methods)
        if method not in methods:
            logging.error(assert_message)
            raise AssertionError(assert_message)