import os.path
from dataclasses import dataclass
from dacite import from_dict, Config
from yaml import safe_load
from pathlib import Path


@dataclass
class Features:
    state: str
    account_length: int
    area_code: str
    international_plan: bool
    voice_mail_plan: bool
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: int
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: int
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: int
    total_intl_charge: float
    number_customer_service_calls: int

@dataclass
class ImpFeatures:
    state: str
    international_plan: bool
    voice_mail_plan: bool
    number_vmail_messages: int
    number_customer_service_calls: int
    total_day_minutes: float
    total_eve_minutes: float
    total_night_minutes: float
    total_day_charge: float
    total_eve_charge: float
    total_night_charge: float


@dataclass
class BackendConfig:
    target_name: str
    classifiers: list[str]
    port: int
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

conf = BackendConfig.from_yaml()