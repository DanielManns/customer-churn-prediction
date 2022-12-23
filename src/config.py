import src
from dataclasses import dataclass
from src.backend.models import config as m_config
from src.backend.utility import config as u_config


@dataclass
class Configuration:
    m_config: src.backend.models.config.Configuration
    u_config: src.backend.utility.config.Configuration


def config():
    return Configuration(m_config=m_config(), u_config=u_config())
