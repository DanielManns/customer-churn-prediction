import os.path
from dataclasses import dataclass
from dacite import from_dict
from yaml import safe_load


@dataclass
class Configuration:
    target_name: str
    im_vars: list[str]
    classifiers: list[str]
    ran_classifiers: list[str]
    train_seed: int


with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as p:
    data = safe_load(p)

config = from_dict(data_class=Configuration, data=data)
