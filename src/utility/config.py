from dataclasses import dataclass


@dataclass
class Configuration:
    data_dir: str
    model_dir: str
    exp_dir: str
    train_data_path: str
    experiments: list[str]
