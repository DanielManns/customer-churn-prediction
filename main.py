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
    arg("-experiment", type=str, default="./experiments/test_experiment.yaml", help="Path to experiment configs")
    arg("--from-checkpoint", dest="checkpoint", action="store_true",
        help="Continue from checkpoint if set")

    args = parser.parse_args()

    with open("config.yaml") as p:
        config = yaml.safe_load(p)

    train_path = config["paths"]["train_path"]
    target_name = config["target_name"]
    im_vars = config["important_vars"]
    train_seed = config["seeds"]["train"]

    raw = pd.read_csv(train_path)

    df = apply_preprocessing(raw)
    df = apply_feature_engineering(df)

    target = df[target_name]
    df = df.drop(columns=[target_name])

    # subset important variables
    # data = data.loc[:, im_vars]

    with open(args.experiment) as p:
        exp_config = yaml.safe_load(p)

    classifiers = exp_config["models"]["class_names"]
    cv_method = eval(exp_config["cross_validation"]["method"])
    col_transformer = create_col_transformer(df)

    for c in classifiers:
        classifier_args = exp_config["classifier_args"][c]
        classifier = eval(c)(**classifier_args) if classifier_args is not None else eval(c)()
        cv_train_pipeline(create_pipeline(col_transformer, classifier), df, target, cv_method, folds=10)
