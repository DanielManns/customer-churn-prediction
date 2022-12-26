from dataclasses import dataclass
from src.backend.src import config as back_config
from src.frontend.src import config as front_config


@dataclass
class Configuration:
    back_config: back_config.BackendConfig
    front_config: front_config.Configuration


def config():
    return Configuration(back_config=back_config.config(), front_config=front_config.config())
