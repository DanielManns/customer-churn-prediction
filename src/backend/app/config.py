import os.path
from dataclasses import dataclass
from dacite import from_dict, Config
from yaml import safe_load
from pathlib import Path


@dataclass
class Configuration:
    target_name: str
    im_vars: list[str]
    classifiers: list[str]
    data_dir: Path
    exp_dir: Path
    train_path: Path
    test_path: Path


def config():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as p:
        data = safe_load(p)

    converters = {
        Path: Path
    }

    return from_dict(data_class=Configuration, data=data, config=Config(type_hooks=converters))