import src
from dataclasses import dataclass
import src.models.config as m_config
import src.utility.config as u_config


@dataclass
class Configuration:
    m_config: src.models.config.Configuration
    u_config: src.utility.config.Configuration


config = Configuration(m_config=m_config.config, u_config=u_config.config)