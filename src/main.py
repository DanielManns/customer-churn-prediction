import pandas as pd
import argparse
import os
from src.utility.argument_parser import parse

import yaml
import warnings
from config import config

from src.models.feature_engineering import apply_feature_engineering
from plotting import plot_feature_importance
from src.models.preprocessing import apply_preprocessing, create_col_transformer, get_cat_features, get_con_features
from src.models.training import create_pipeline, train_pipeline, cv_pipeline, get_feature_importance

from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold

pd.set_option('display.max_columns', 500)
c = config()


def warn(*args, **kwargs):
    pass


def run_experiment(mixed_df: pd.DataFrame, exp_config: dict) -> None:
    """
    Runs experiment with given experiment dict, already loaded from .yaml.

    :param mixed_df: Pandas DataFrame - Cleaned and enriched data (all features)
    :param exp_config: dict - experiment dictionary
    :return: None
    """

    is_subset = exp_config["features"]["is_subset"]

    # subset important variables
    if is_subset:
        mixed_df = mixed_df.loc[:, os.environ["IM_VARS"]]

    cat_df = mixed_df.drop(columns=get_con_features(mixed_df))
    con_df = mixed_df.drop(columns=get_cat_features(mixed_df))

    classifiers = exp_config["classifiers"]
    cv_method = eval(exp_config["cross_validation"]["class_name"])
    cv_method_kwargs = exp_config["cross_validation"]["params"]

    for _, cl in classifiers.items():
        c_params = cl["params"]
        # append training seed if classifiers has random component
        if cl["class_name"] in c.m_config.ran_classifiers:
            c_params = {**c_params, **{"random_state": c.m_config.train_seed}}
        classifier = eval(cl["class_name"])(**c_params) if c_params is not None else eval(cl["class_name"])()
        if cl["type"] == "categorical":
            X = cat_df
        elif cl["type"] == "continuous":
            X = con_df
        else:
            X = mixed_df

        col_transformer = create_col_transformer(X)
        pipe = create_pipeline(col_transformer, classifier)
        train_pipeline(pipe, X, y)
        cv_pipelines = cv_pipeline(pipe, X, y, cv_method(**cv_method_kwargs))
        if classifier.__class__.__name__ in ["DecisionTreeClassifier", "LogisticRegression"]:
            plot_feature_importance(get_feature_importance(pipe))
        print("\n")


if __name__ == "__main__":
    args = parse()
    print(args)

    # supress warnings
    warnings.warn = warn

    raw = pd.read_csv(c.u_config.train_path)

    mixed_df = apply_preprocessing(raw)
    mixed_df = apply_feature_engineering(mixed_df)

    y = mixed_df[c.m_config.target_name]
    mixed_df = mixed_df.drop(columns=[c.m_config.target_name])

    for exp_name in c.u_config.experiments:
        exp_path = os.path.join(c.u_config.exp_dir, exp_name)
        print(f"\nRunning experiment located at {exp_path} ...")
        with open(exp_path) as p:
            exp_config = yaml.safe_load(p)
        run_experiment(mixed_df, exp_config)


