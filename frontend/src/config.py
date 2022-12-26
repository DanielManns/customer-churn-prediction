import os.path
from dataclasses import dataclass
from dacite import from_dict, Config
from yaml import safe_load
from pathlib import Path


@dataclass
class Configuration:
    test: str


def config():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as p:
        data = safe_load(p)

    converters = {
        Path: Path
    }

    return from_dict(data_class=Configuration, data=data, config=Config(type_hooks=converters))