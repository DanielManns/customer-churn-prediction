import src
from dataclasses import dataclass
from src.backend.ml import config as m_config
from src.backend.utility import config as u_config


@dataclass
class Configuration:
    m_config: m_config.Configuration
    u_config: u_config.Configuration


def config():
    return Configuration(m_config=m_config.config(), u_config=u_config.config())
