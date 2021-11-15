import os
import logging

class AssertConditionConstants:
    languages = ["eng", "kor"]
    available_models = ["transformer", "bert", "poly-encoder", "gpt2"]
    available_optimizers = ["adam", "adam_w"]
    preprocess_approaches = ["stop", "ignore", "truncate"]
    aggregation_methods = ["first", "last", "sum", "average"]
    decoding_methods = ["greedy", "beam_search", "top_k_sampling"]
    metrics = ["bleu", "meteor", "rouge", "hits", "semantic_score"]

class Assertion:
    def assert_isinstance_list(self, data, parameter_name):
        assert_message = "The data type of parameter '{parameter}' must be list".format(parameter=parameter_name)
        if not isinstance(data, list):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isinstance_dict(self, data, parameter_name):
        assert_message = "The data type of parameter '{parameter}' must be dict".format(parameter=parameter_name)
        if not isinstance(data, dict):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isintance(self, obj, data_type):
        assert_message = "The data type of parameter 'obj' must be {data_type}".format(data_type=data_type.__name__)
        if not isinstance(obj, data_type):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isequal_dict(self, a, b):
        assert_message = "Given two dictionaries must be equal: {a} vs {b}".format(a=a, b=b)
        if a != b:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isequal_elements_length(self, data):
        assert_message = "All elements of data must have equal length: {data_length} vs {element_length}"
        length = len(data[0])
        for element in data:
            if len(element) != length:
                assert_message = assert_message.format(data_length=length, element_length=len(element))
                logging.error(assert_message)
                raise AssertionError(assert_message)

    def assert_contain_elements(self, required, target, name=None):
        assert_message = "Data must contain following element: {element}"
        if name is not None: assert_message = "{name} must contain following element: '{{element}}'".format(name=name)
        for element in required:
            if element not in target:
                assert_message = assert_message.format(element=element)
                logging.error(assert_message)
                raise AssertionError(assert_message)

    def assert_isequal_keys(self, a, b, except_keys=None):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        if except_keys is not None:
            a_keys = a_keys.difference(set(except_keys))
            b_keys = b_keys.difference(set(except_keys))
        assert_message = "The keys of two dictionaries must be equal: {a_keys} vs {b_keys}".format(a_keys=a_keys, b_keys=b_keys)
        if a_keys != b_keys:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_implemented(self, method_name):
        assert_message = "{method_name} method must be implemented".format(method_name=method_name)
        logging.error(assert_message)
        raise AssertionError(assert_message)

    def assert_equal(self, a, b):
        assert_message = "Given inputs must be equal: {a} vs {b}".format(a=a, b=b)
        if a != b:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_equal_length(self, a, b):
        assert_message = "The length of given inputs must be equal: {len1} vs {len2}".format(len1=len(a), len2=len(b))
        if len(a) != len(b):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_equal_or_greater(self, value, criteria):
        assert_message = "Given value must be equal or greater than {criteria}".format(criteria=criteria)
        if value < criteria:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_equal_or_lesser(self, value, criteria):
        assert_message = "Given value must be equal or lesser than {criteria}".format(criteria=criteria)
        if value > criteria:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_greater_length(self, data, length):
        assert_message = "The length of given data must be greater than {length}".format(length=length)
        if len(data) <= length:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_lesser_length(self, data, length):
        assert_message = "The length of given data must be lesser than {length}".format(length=length)
        if len(data) >= length:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_is_not_none(self, obj, name=None):
        assert_message = "Object must not be None"
        if name is not None: assert_message = "'{name}' must not be None".format(name=name)
        if obj is None:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_not_out_of_index(self, index, upperbound):
        assert_message = "Index must be less than length {upperbound}".format(upperbound=upperbound)
        if index >= upperbound:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_isin_obj(self, element, obj):
        assert_message = "Element must be one of elements of obj: {given} is not in {obj}".format(given=element, obj=obj)
        if element not in obj:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_proper_extension(self, path, extensions):
        assert_message = "Path must ends with '{extension}': {path}".format(extension=extensions, path=path)
        flag = False
        for extension in extensions:
            if path.endswith(extension):
                flag = True
                break

        if not flag:
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_is_valid_file(self, path):
        assert_message = "Invalid or File not exists: {path}".format(path=path)
        if not os.path.exists(path) and not os.path.isfile(path):
            logging.error(assert_message)
            raise AssertionError(assert_message)

    def assert_is_valid_path(self, path):
        assert_message = "Invalid or Path not exists: {path}".format(path=path)
        if not os.path.exists(path) and not os.path.isdir(path):
            logging.error(assert_message)
            raise AssertionError(assert_message)