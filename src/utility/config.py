from dataclasses import dataclass
from pathlib import Path
from yaml import safe_load
from dacite import from_dict, Config
import os


@dataclass
class Configuration:
    data_dir: Path
    model_dir: Path
    exp_dir: Path
    train_path: Path
    experiments: list[str]


def config():
    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as p:
        data = safe_load(p)

    converters = {
        Path: Path
    }

    return from_dict(data_class=Configuration, data=data, config=Config(type_hooks=converters))
