import os.path
from dataclasses import dataclass
from dacite import from_dict, Config
from yaml import safe_load
from pathlib import Path


@dataclass
class BackendConfig:
    target_name: str
    im_vars: list[str]
    classifiers: list[str]
    port: 8000
    host: str
    reload: bool

    @classmethod
    def from_yaml(cls):
        with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as p:
            data = safe_load(p)

        converters = {
            Path: Path
        }

        return from_dict(data_class=BackendConfig, data=data, config=Config(type_hooks=converters))