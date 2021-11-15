import os
import json
import logging
from datetime import datetime

path = "./config/logging_config.json"
config = None
with open(path, encoding="UTF-8") as fp:
    config = json.load(fp)
log_file_path = config["log_filepath"]
log_file_dir = "/".join(log_file_path.split("/")[:-1])
if not os.path.exists(log_file_dir): os.makedirs(log_file_dir)

today_date_str = datetime.today().date().strftime("%Y%m%d")
_log_file_path = log_file_path.format(name="main", date=today_date_str)
logging.basicConfig(level=logging.INFO, format=config["format"], filename=_log_file_path, filemode="a")


def get_logger(name):
    # create or get logger
    logger = logging.getLogger(name)

    # set log level
    level = logging.INFO
    if config["level"]=="CRITICAL":
        level = logging.CRITICAL
    elif config["level"]=="ERROR":
        level = logging.ERROR
    elif config["level"]=="WARNING":
        level = logging.WARNING
    elif config["level"]=="INFO":
        level = logging.INFO
    elif config["level"]=="DEBUG":
        level = logging.DEBUG
    elif config["level"]=="NOTSET":
        level = logging.NOTSET
    logger.setLevel(level)

    # define file handler and set formatter
    today_date_str = datetime.today().date().strftime("%Y%m%d")
    _log_file_path = log_file_path.format(name=name, date=today_date_str)
    file_handler = logging.FileHandler(_log_file_path, mode=config["filemode"])
    formatter = logging.Formatter(config["format"])
    file_handler.setFormatter(formatter)
    # add file handler to logger
    logger.addHandler(file_handler)

    return logger