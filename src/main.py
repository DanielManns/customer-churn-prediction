import pandas as pd
import argparse
import os

import yaml
import warnings

from src.models.feature_engineering import apply_feature_engineering
from plotting import plot_feature_importance
from src.models.preprocessing import apply_preprocessing, create_col_transformer, get_cat_features, get_con_features
from src.models.training import create_pipeline, train_pipeline, cv_pipeline, get_feature_importance

pd.set_option('display.max_columns', 500)


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

    for _, c in classifiers.items():
        c_params = c["params"]
        # append training seed if classifiers has random component
        if c["class_name"] in ran_classifiers:
            c_params = {**c_params, **{"random_state": os.environ["TRAIN_SEED"]}}
        classifier = eval(c["class_name"])(**c_params) if c_params is not None else eval(c["class_name"])()
        if c["type"] == "categorical":
            X = cat_df
        elif c["type"] == "continuous":
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

    # supress warnings
    warnings.warn = warn

    # TODO refactor this
    with open("../config.yaml") as p:
        config = yaml.safe_load(p)

    os.environ["TRAIN_PATH"] = config["paths"]["train_path"]
    os.environ["TARGET_NAME"] = config["target_name"]

    os.environ["TRAIN_SEED"] = str(config["seeds"]["train"])

    ran_classifiers = config["random_classifiers"]
    im_vars = config["important_vars"]

    raw = pd.read_csv(os.environ["TRAIN_PATH"])

    mixed_df = apply_preprocessing(raw)
    mixed_df = apply_feature_engineering(mixed_df)

    y = mixed_df[os.environ["TARGET_NAME"]]
    mixed_df = mixed_df.drop(columns=[os.environ["TARGET_NAME"]])

    for exp_name in config["experiments"]:
        exp_path = os.path.join(config["dirs"]["exp_dir"], exp_name)
        print(f"\nRunning experiment located at {exp_path} ...")
        with open(exp_path) as p:
            exp_config = yaml.safe_load(p)
        run_experiment(mixed_df, exp_config)


