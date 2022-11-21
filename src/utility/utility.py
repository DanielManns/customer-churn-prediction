import os
import yaml
import config
import pandas as pd
from sklearn.base import ClassifierMixin
import pickle
import src.models.preprocessing as pre

con = config.config()


def load_raw_data() -> pd.DataFrame:
    """
    Returns raw DataFrame.

    :return: pd.DataFrame - raw data
    """

    return pd.read_csv(con.u_config.train_path)


def load_exp_config(exp_name) -> dict:
    """
    Returns experimental config from given experiment path.

    :param exp_name: str - experiment path
    :return: dict - experiment config
    """

    exp_path = get_exp_conf_path(exp_name)

    with open(exp_path, "rb") as p:
        print(f"\nLoaded experiment located at {exp_path} ...")
        return yaml.safe_load(p)


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

    clf_path = get_clf_path(exp_name, clf.__class__.__name__)
    pickle.dump(clf, open(clf_path, "wb"))


def load_clf(exp_name: str, clf: str) -> ClassifierMixin:
    """
    Loads classifiers latest checkpoint.

    :param exp_name: str - experiment name
    :param clf: ClassifierMixin - classifier
    :return: ClassifierMixin - loaded classifier
    """

    clf_path = get_clf_path(exp_name, clf.__class__.__name__)
    return pickle.load(open(clf_path, "rb"))


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


def get_exp_plot_dir(exp_name: str) -> str:
    """
    Returns postprocessing directory for given experiment name.

    :param exp_name: str - experiment name
    :return: str - postprocessing dir
    """

    return os.path.join(get_exp_dir(exp_name), "plots")


def get_exp_conf_path(exp_name) -> str:
    """
    Returns experiment configuration path from given experiment name.

    :param exp_name: str- experiment name
    :return: str - experiment configuration path
    """

    return os.path.join(get_exp_dir(exp_name), exp_name + ".yaml")


def get_clf_path(exp_name, clf_class_name):
    """
    Returns classifier checkpoint path for given experiment name and classifier.

    :param exp_name: str - experiment name
    :param clf_class_name: ClassifierMixin - classifier
    :return: str - checkpoint path
    """

    return os.path.join(get_exp_check_dir(exp_name), clf_class_name + ".sav")


def create_exp_dirs(exp_name: str) -> None:
    """
    Creates required directories for a given experiment name
    :param exp_name: str - experiment name
    :return: None
    """

    exp_dir = get_exp_dir(exp_name)
    exp_check_dir = get_exp_check_dir(exp_name)
    dirs = [exp_dir, exp_check_dir]

    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d)


def create_pp_dirs(exp_name: str) -> None:
    """
    Creates required postprocessing directories for a given experiment name.
    :param exp_name: str - experiment name
    :return: None
    """

    plot_dir = get_exp_plot_dir(exp_name)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)




