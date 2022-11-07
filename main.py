import numpy as np
import pandas as pd
import argparse
import os
from IPython.display import display, HTML

from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
import yaml

from feature_engineering import apply_feature_engineering
from preprocessing import apply_preprocessing, create_col_transformer, get_cat_features, get_con_features
from training import create_pipeline, train_pipeline, cv_train_pipeline

pd.set_option('display.max_columns', 500)


def run_experiment(mixed_df: pd.DataFrame, exp_kwargs: dict) -> None:
    """
    Runs experiment with given experiment dict, already loaded from .yaml.

    :param mixed_df: Pandas DataFrame - Cleaned and enriched data (all features)
    :param exp_kwargs: dict - experiment dictionary
    :return: None
    """

    is_subset = exp_config["features"]["is_subset"]

    # subset important variables
    if is_subset:
        mixed_df = mixed_df.loc[:, im_vars]

    cat_df = mixed_df.drop(columns=get_con_features(mixed_df))
    con_df = mixed_df.drop(columns=get_cat_features(mixed_df))

    classifiers = exp_config["classifiers"]
    cv_method = eval(exp_config["validation"]["method"])

    for _, c in classifiers.items():
        c_params = c["params"]
        # append training seed if classifiers has random component
        if c["class_name"] in random_cs:
            c_params = {**c_params, **{"random_state": train_seed}}
        classifier = eval(c["class_name"])(**c_params) if c_params is not None else eval(c["class_name"])()
        if c["type"] == "categorical":
            df = cat_df
        elif c["type"] == "continuous":
            df = con_df
        else:
            df = mixed_df

        col_transformer = create_col_transformer(df)
        cv_train_pipeline(create_pipeline(col_transformer, classifier), df, target, cv_method, folds=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict whether telecommunication customers churned')
    arg = parser.add_argument
    arg("-c", "--classifiers", nargs="+", help="<Required> A minimum of 1 classifiers (sklearn class names)",
        required=False)
    arg("-p", "--classifier_params", nargs="+", help="<Required> A minimum of 1 classifier parameters", required=False)
    arg("-v", "--validation", type=str, default="cross_validate", help="Cross validation method (sklearn class name)")
    # arg("-s", "--is_subset", action="store_true", help="Only use important features if set")
    arg("-e", "--exp_name", type=str, default="test_experiment.yaml", help="Path to experiment configs")

    args = parser.parse_args()
    print(args)

    with open("config.yaml") as p:
        config = yaml.safe_load(p)

    train_path = config["paths"]["train_path"]
    target_name = config["target_name"]
    im_vars = config["important_vars"]
    train_seed = config["seeds"]["train"]
    random_cs = config["random_classifiers"]

    raw = pd.read_csv(train_path)

    mixed_df = apply_preprocessing(raw)
    mixed_df = apply_feature_engineering(mixed_df)

    target = mixed_df[target_name]
    mixed_df = mixed_df.drop(columns=[target_name])

    for exp_name in config["experiments"]:
        exp_path = os.path.join(config["dirs"]["exp_dir"], exp_name)
        print(f"\nRunning experiment located at {exp_path} ...")
        with open(exp_path) as p:
            exp_config = yaml.safe_load(p)
        run_experiment(mixed_df, exp_config)


