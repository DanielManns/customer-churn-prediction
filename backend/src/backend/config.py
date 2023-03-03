import os.path
from dataclasses import dataclass
from dacite import from_dict, Config
from yaml import safe_load
from pathlib import Path
from typing import Union
from pydantic import BaseModel
from enum import Enum

class ExpName(str, Enum):
    exp_no_subset = "exp_no_subset"
    exp_subset = "exp_subset"

class Row(BaseModel):
    id: Union[int, None] = None
    state: Union[str, None] = None
    account_length: Union[int, None] = None
    area_code: Union[str, None] = None
    international_plan: Union[bool, None] = None
    voice_mail_plan: Union[bool, None] = None
    number_vmail_messages: Union[int, None] = None
    total_day_minutes: Union[float, None] = None
    total_day_calls: Union[int, None] = None
    total_day_charge: Union[float, None] = None
    total_eve_minutes: Union[float, None] = None
    total_eve_calls: Union[int, None] = None
    total_eve_charge: Union[float, None] = None
    total_night_minutes: Union[float, None] = None
    total_night_calls: Union[int, None] = None
    total_night_charge: Union[float, None] = None
    total_intl_minutes: Union[float, None] = None
    total_intl_calls: Union[int, None] = None
    total_intl_charge: Union[float, None] = None
    number_customer_service_calls: Union[int, None] = None

class PredRow(BaseModel):
    id: Union[int, None] = None
    pred: Union[float, None] = None

class ImportanceRow(BaseModel):
    state: Union[str, None] = None
    account_length: Union[int, None] = None
    area_code: Union[str, None] = None
    international_plan: Union[bool, None] = None
    voice_mail_plan: Union[bool, None] = None
    number_vmail_messages: Union[int, None] = None
    total_day_minutes: Union[float, None] = None
    total_day_calls: Union[int, None] = None
    total_day_charge: Union[float, None] = None
    total_eve_minutes: Union[float, None] = None
    total_eve_calls: Union[int, None] = None
    total_eve_charge: Union[float, None] = None
    total_night_minutes: Union[float, None] = None
    total_night_calls: Union[int, None] = None
    total_night_charge: Union[float, None] = None
    total_intl_minutes: Union[float, None] = None
    total_intl_calls: Union[int, None] = None
    total_intl_charge: Union[float, None] = None
    number_customer_service_calls: Union[int, None] = None
    total_reg_charge: Union[float, None] = None
    total_reg_calls: Union[int, None] = None
    total_reg_minutes: Union[float, None] = None
    avg_intl_call_duration: Union[float, None] = None
    avg_day_call_duration: Union[float, None] = None
    avg_eve_call_duration: Union[float, None] = None
    avg_night_call_duration: Union[float, None] = None

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