import numpy as np
import pandas as pd
import argparse
from IPython.display import display, HTML

from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
import yaml

from feature_engineering import apply_feature_engineering
from preprocessing import apply_preprocessing, create_col_transformer
from training import create_pipeline, train_pipeline, cv_train_pipeline

pd.set_option('display.max_columns', 500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict whether telecommunication customers churned')
    arg = parser.add_argument
    arg("-c", "--classifiers", nargs="+", help="<Required> A minimum of 1 classifiers (sklearn class names)", required=False)
    arg("-p", "--classifier_params", nargs="+", help="<Required> A minimum of 1 classifier parameters", required=False)
    arg("-v", "--validation", type=str, default="cross_validate", help="Cross validation method (sklearn class name)")
    arg("-s", "--subset", dest="is_subset", action="store_true",
        help="Only use important features if set")
    arg("-e", "--exp_path", type=str, default="./experiments/test_experiment.yaml", help="Path to experiment configs")

    args = parser.parse_args()

    with open("config.yaml") as p:
        config = yaml.safe_load(p)

    train_path = config["paths"]["train_path"]
    target_name = config["target_name"]
    im_vars = config["important_vars"]
    train_seed = config["seeds"]["train"]
    random_cs = config["random_classifiers"]

    raw = pd.read_csv(train_path)

    df = apply_preprocessing(raw)
    df = apply_feature_engineering(df)

    target = df[target_name]
    df = df.drop(columns=[target_name])

    # subset important variables
    # data = data.loc[:, im_vars]

    with open(args.exp_path) as p:
        exp_config = yaml.safe_load(p)

    classifiers = exp_config["classifiers"]["class_names"]
    cv_method = eval(exp_config["validation"]["method"])
    col_transformer = create_col_transformer(df)

    for c_name in classifiers:
        classifier_args = exp_config["classifier_params"][c_name]
        # append training seed if classifiers has random component
        if c_name in random_cs:
            classifier_args = {**classifier_args, **{"random_state": train_seed}}
        classifier = eval(c_name)(**classifier_args) if classifier_args is not None else eval(c_name)()
        cv_train_pipeline(create_pipeline(col_transformer, classifier), df, target, cv_method, folds=10)
