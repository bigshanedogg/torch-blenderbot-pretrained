from transformer.services.dialog_generator.gpt2 import Gpt2DialogGenerator
from transformer.services.dialog_generator.bart import BartDialogGenerator
from transformer.services.dialog_retriever.poly_encoder import PolyEncoderDialogRetriever

def get_dialog_generators(model_dir, model_path_config, device):
    dialog_generators = dict()
    dialog_generators["gpt2-dev"] = Gpt2DialogGenerator()
    dialog_generators["bart-dev"] = BartDialogGenerator()
    for version, _version_config in model_path_config["dialog-generator"].items():
        _model = _version_config["model"]
        _path = model_dir + _version_config["path"]

        print("# loading '{module}/{version}' from '{path}'".format(module="dialog-generator", version=version, path=_path))
        dialog_generator = None
        if _model == "gpt2": dialog_generator = Gpt2DialogGenerator()
        elif _model == "bart": dialog_generator = BartDialogGenerator()
        dialog_generator.set_device(device=device)
        dialog_generator.load_model(model_dir=_path)
        dialog_generators[version] = dialog_generator
    return dialog_generators

def get_dialog_retrievers(model_dir, model_path_config, device):
    dialog_retrievers = dict()
    dialog_retrievers["poly-encoder-dev"] = PolyEncoderDialogRetriever()
    for version, _version_config in model_path_config["dialog-retriever"].items():
        _model = _version_config["model"]
        _path = model_dir + _version_config["path"]

        print("# loading '{module}/{version}' from '{path}'".format(module="dialog-retriever", version=version, path=_path))
        dialog_retriever = None
        if _version_config["model"] == "poly-encoder": dialog_retriever = PolyEncoderDialogRetriever()
        dialog_retriever.set_device(device=device)
        dialog_retriever.load_model(model_dir=_path)
        dialog_retrievers[version] = dialog_retriever
    return dialog_retrievers

def get_dialog_blenders():
    pass