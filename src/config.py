import src
from dataclasses import dataclass
from src.models.config import config as m_config
from src.utility.config import config as u_config


@dataclass
class Configuration:
    m_config: src.models.config.Configuration
    u_config: src.utility.config.Configuration


def config():
    return Configuration(m_config=m_config(), u_config=u_config())
