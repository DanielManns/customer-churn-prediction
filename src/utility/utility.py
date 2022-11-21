import os
import yaml
import config
import pandas as pd
from sklearn.base import ClassifierMixin
import pickle
import src.models.preprocessing as pre

con = config.config()


def load_exp_models(exp_name: str) -> [[ClassifierMixin], [pd.DataFrame], pd.DataFrame]:
    """
    Loads already trained models for given experimental config.

    :param exp_name: str - experiment name
    :return:
    """

    exp_path = get_exp_path(exp_name)
    exp_config = get_exp_config(exp_path)
    cat_X, con_X, mixed_X, y = pre.get_exp_dfs(exp_config)
    classifiers = exp_config["classifiers"]
    loaded_clfs = []
    dfs = []
    filename = exp_config["checkpoint_path"]

    for _, c in classifiers.items():
        loaded_clfs.append(pickle.load(open(filename, "rb")))
        if c["type"] == "categorical":
            X = cat_X
        elif c["type"] == "continuous":
            X = con_X
        else:
            X = mixed_X
        dfs.append(X)

    return loaded_clfs, dfs, y


def get_raw_data() -> pd.DataFrame:
    """
    Returns raw DataFrame.

    :return: pd.DataFrame - raw data
    """

    return pd.read_csv(con.u_config.train_path)


def get_exp_path(exp_name) -> str:
    """
    Returns experiment path from given experiment name.

    :param exp_name: str- experiment name
    :return: str - experiment path
    """

    dir = os.path.join(con.u_config.exp_dir, exp_name)
    return os.path.join(dir, exp_name + ".yaml")


def get_exp_config(exp_path) -> dict:
    """
    Returns experimental config from given experiment path.

    :param exp_path: str - experiment path
    :return: dict - experiment config
    """

    with open(exp_path) as p:
        return yaml.safe_load(p)