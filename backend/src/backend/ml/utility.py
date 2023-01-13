import os
from importlib.abc import Traversable

import yaml
from sklearn.compose import ColumnTransformer
import backend.data as data
from backend.data import experiments
from backend.config import BackendConfig
import pandas as pd
from sklearn.base import ClassifierMixin
import pickle
from importlib import resources

conf = BackendConfig.from_yaml()


def load_train_dataset() -> pd.DataFrame:
    """
    Returns raw train DataFrame.

    :return: pd.DataFrame - raw train data
    """

    return pd.read_csv(resources.files(data).joinpath("train.csv"))


def load_test_dataset() -> pd.DataFrame:
    """
    Returns raw test DataFrame.

    :return: pd.DataFrame - raw test data
    """
    return pd.read_csv(resources.files(data).joinpath("test.csv"), index_col="id")


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


def save_clf(exp_name: str, clf: ClassifierMixin, i: int):
    """
    Saves checkpoint of given classifier.

    :param exp_name: str - experiment name
    :param clf: ClassifierMixin - classifier
    :param i: iteration of the same experiment
    :return: None
    """

    clf_path = get_clf_path(exp_name, clf.__class__.__name__, i)
    pickle.dump(clf, open(clf_path, "wb"))


def save_clfs(exp_name: str, clfs: list[ClassifierMixin]):
    """
    Saves list of classifiers.

    :param exp_name: str - experiment name
    :param clfs: [ClassifierMixin] - classifier list
    :return:
    """

    for i, clf in enumerate(clfs):
        save_clf(exp_name, clf, i)


def save_scaler(exp_name: str, scaler: ColumnTransformer):
    """
    Saves scaler for given experiment.

    :param exp_name: str- experiment name
    :param scaler: ColumnTransformer - scaler
    :return: None
    """
    scaler_path = get_scaler_path(exp_name)
    pickle.dump(scaler, open(scaler_path, "wb"))


def load_clf(exp_name: str, clf_class: str, i: int) -> ClassifierMixin:
    """
    Loads classifiers from the latest checkpoint.

    :param exp_name: str - experiment name
    :param clf_class: str - classifier class name
    :param i: experiment iteration
    :return: ClassifierMixin - loaded classifier
    """

    clf_path = get_clf_path(exp_name, clf_class, i)
    return pickle.load(open(clf_path, "rb"))


def load_cv_clfs(exp_name: str, clf_class: str, n_splits: int) -> list[ClassifierMixin]:
    """
    Loads the latest checkpoint of all classifiers of given class.
    :param exp_name: str - experiment name
    :param clf_class: str - classifier class name
    :param n_splits: int - number of folds for cross validation
    :return: [ClassifierMixin] - List of trained classifiers
    """

    clfs = []
    for i in range(n_splits):
        clfs.append(load_clf(exp_name, clf_class, i))
    return clfs


def load_scaler(exp_name: str) -> ColumnTransformer:
    """
    Loads the trained scaler for given experiment.

    :param exp_name: str - experiment name
    :return: ColumnTransformer - scaler
    """

    scaler_path = get_scaler_path(exp_name)
    return pickle.load(open(scaler_path, "rb"))


def get_exp_dir(exp_name: str) -> Traversable:
    """
    Returns experiment dit with given experiment name.

    :param exp_name: str - experiment name
    :return: str - experiment dir
    """

    return resources.files(experiments).joinpath(exp_name)


def get_exp_check_dir(exp_name: str) -> Traversable:
    """
    Returns checkpoint dir for given experiment name.
    :param exp_name: str - experiment name
    :return: str - checkpoint dir
    """

    return get_exp_dir(exp_name).joinpath("checkpoints")


def get_exp_plot_dir(exp_name: str) -> Traversable:
    """
    Returns postprocessing directory for given experiment name.

    :param exp_name: str - experiment name
    :return: str - postprocessing dir
    """

    return get_exp_dir(exp_name).joinpath("plots")


def get_exp_conf_path(exp_name) -> Traversable:
    """
    Returns experiment configuration path from given experiment name.

    :param exp_name: str- experiment name
    :return: str - experiment configuration path
    """

    return get_exp_dir(exp_name).joinpath(exp_name + ".yaml")


def get_clf_path(exp_name, clf_class_name, i: int) -> Traversable:
    """
    Returns classifier checkpoint path for given experiment name and classifier.

    :param exp_name: str - experiment name
    :param clf_class_name: ClassifierMixin - classifier
    :param i: int - iteration of the same experiment
    :return: str - checkpoint path
    """

    return get_exp_check_dir(exp_name).joinpath(clf_class_name + "_" + str(i) + ".sav")


def get_scaler_path(exp_name: str) -> Traversable:
    """
    Returns scaler path for given experiment name.

    :param exp_name: str - experiment name
    :return: str - scaler path
    """

    return get_exp_check_dir(exp_name).joinpath("scaler.sav")
