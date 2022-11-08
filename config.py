from dataclasses import dataclass

import src.models.config


@dataclass
class Configuration:
    version: 1
    utility: src.utility.config.Configuration
    models: src.models.config.Configuration
