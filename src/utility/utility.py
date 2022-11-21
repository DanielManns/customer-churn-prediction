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

    exp_path = get_exp_conf_path(exp_name)
    exp_config = load_exp_config(exp_path)
    cat_X, con_X, mixed_X, y = pre.get_exp_df(exp_config)
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


def save_clf(exp_name: str, clf: ClassifierMixin):
    """
    Saves checkpoint of given classifier.

    :param exp_name: str - experiment name
    :param clf: ClassifierMixin - classifier
    :return: None
    """

    clf_path = get_clf_path(exp_name, clf)
    pickle.dump(clf, open(clf_path, "wb"))


def load_clf(exp_name: str, clf: ClassifierMixin) -> ClassifierMixin:
    """
    Loads classifiers latest checkpoint.

    :param exp_name: str - experiment name
    :param clf: ClassifierMixin - classifier
    :return: ClassifierMixin - loaded classifier
    """

    clf_path = get_clf_path(exp_name, clf)
    return pickle.load(open(clf_path, "rb"))

def get_raw_data() -> pd.DataFrame:
    """
    Returns raw DataFrame.

    :return: pd.DataFrame - raw data
    """

    return pd.read_csv(con.u_config.train_path)


def get_exp_dir(exp_name: str) -> str:
    """
    Returns experiment dit with given experiment name.

    :param exp_name: str - experiment name
    :return: str - experiment dir
    """

    return os.path.join(con.u_config.exp_dir, exp_name)


def get_exp_check_dir(exp_name: str) -> str:
    """
    Returns checkpoint dir for given experiment name.
    :param exp_name: str - experiment name
    :return: str - checkpoint dir
    """

    return os.path.join(get_exp_dir(exp_name), "checkpoints")


def get_exp_conf_path(exp_name) -> str:
    """
    Returns experiment configuration path from given experiment name.

    :param exp_name: str- experiment name
    :return: str - experiment configuration path
    """

    return os.path.join(get_exp_conf_path(exp_name), exp_name + ".yaml")


def get_clf_path(exp_name, clf):
    """
    Returns classifier checkpoint path for given experiment name and classifier.

    :param exp_name: str - experiment name
    :param clf: ClassifierMixin - classifier
    :return: str - checkpoint path
    """

    return os.path.join(get_exp_check_dir(exp_name), clf.__class__.__name__)


def load_exp_config(exp_path) -> dict:
    """
    Returns experimental config from given experiment path.

    :param exp_path: str - experiment path
    :return: dict - experiment config
    """

    with open(exp_path) as p:
        return yaml.safe_load(p)