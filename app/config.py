from dataclasses import dataclass
from app.backend.app.ml import config as ml_config
from app.backend.app.utility import config as u_config


@dataclass
class Configuration:
    ml_config: ml_config.Configuration
    u_config: u_config.Configuration


def config():
    return Configuration(ml_config=ml_config.config(), u_config=u_config.config())
