import json

def read_json(path, encoding="utf-8"):
    data = None
    with open(path, "r", encoding=encoding) as fp:
        data = json.load(fp)
    return data