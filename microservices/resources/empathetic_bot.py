import json
from microservices.resources.utils import get_dialog_generators, get_dialog_retrievers

model_dir = None
# model_dir = "D:/_jupyter/model"
# model_dir = "/home/ubuntu/data/model"
model_path_config = None
with open("./microservices/model_path_config.json", "r", encoding="utf-8") as fp:
    model_path_config = json.load(fp)
    model_dir = model_path_config["model_dir"]
    if not model_dir.endswith("/"): model_dir += "/"
    model_path_config = model_path_config["empathetic-bot"]

default_device = "cuda:0"
model_path_template = "/{model_path}/"

dialog_generators = get_dialog_generators(model_dir=model_dir, model_path_config=model_path_config, device=default_device)
dialog_retrievers = get_dialog_retrievers(model_dir=model_dir, model_path_config=model_path_config, device=default_device)