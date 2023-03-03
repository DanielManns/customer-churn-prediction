import pandas as pd
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from backend.ml.model import train_model, predict_model, explain_model, visualize_model
from backend.ml.preprocessing import get_train_dataset, scale_df, enrich_df, get_clean_dataset
from backend.config import conf
from backend.ml.utility import load_cv_clfs, save_clfs, save_scaler, load_scaler
from typing import List


CV_METHOD = RepeatedKFold
METRIC = accuracy_score

def train_experiment(exp_config: dict) -> list[list[ClassifierMixin], list[float], list[float]]:
    """
    Runs training for given experimental configuration.

    :param exp_config: dict - experiment configuration
    :return:
    """

    X_clean, y = get_clean_dataset(exp_config)
    
    X_enriched = enrich_df(X_clean)
    X, scaler = scale_df(X_enriched)
    save_scaler(exp_config["name"], scaler)

    classifiers = exp_config["classifiers"]
    cv_method = CV_METHOD(**exp_config["cross_validation"]["params"])

    result = {}

    for _, c in classifiers.items():
        clf = eval(c["class_name"])(**c["params"])

        clfs, train_scores, test_scores = train_model(clf, X, y, cv_method, METRIC)
        save_clfs(exp_config["name"], clfs)
        result[c["class_name"]] = (clfs, train_scores, test_scores)

        print(f"Train accuracy score of {c['class_name']}: {np.array(train_scores).mean()} ± {np.array(train_scores).std()}")
        print(f"Test accuracy score of {c['class_name']}: {np.array(test_scores).mean()} ± {np.array(test_scores).std()}")


    return result


def predict_experiment(exp_config: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs inference for given experiment configuration.

    :param exp_config: dict - experiment configuration
    :param X: pd.DataFrame - clean but not enriched dataframe
    :return: pd.DataFrame - Predictions
    """

    df = enrich_df(df)
    scaler = load_scaler(exp_config["name"])

    X, _ = scale_df(df, scaler)
    
    classifiers = exp_config["classifiers"]
    n_splits = exp_config["cross_validation"]["params"]["n_splits"]

    mean_clf_preds = []
    std_clf_preds = []

    for _, c in classifiers.items():
        clfs = load_cv_clfs(exp_config["name"], c["class_name"], n_splits)
        mean_pred, std_pred = predict_model(clfs, X)
        mean_clf_preds.append(mean_pred)
        std_clf_preds.append(std_pred)

    mean_clf_preds = np.array(mean_clf_preds).T

    data = {"id": df["id"], "DecisionTreeClassifier": mean_clf_preds.squeeze()}

    df = pd.DataFrame(data=data, index=np.arange(df.shape[0]))

    return df


def explain_experiment(exp_config: dict) -> list[pd.DataFrame]:
    classifiers = exp_config["classifiers"]

    scaler = load_scaler(exp_config["name"])
    feature_names = scaler.get_feature_names_out()
    n_splits = exp_config["cross_validation"]["params"]["n_splits"]

    clf_feature_importance = []

    for _, c in classifiers.items():
        clfs = load_cv_clfs(exp_config["name"], c["class_name"], n_splits)
        clf_feature_importance.append(explain_model(clfs, feature_names))

    return clf_feature_importance


def visualize_experiment(exp_config: dict) -> List[List[str]]:
    classifiers = exp_config["classifiers"]
    n_splits = exp_config["cross_validation"]["params"]["n_splits"]
    clf_visualizations = []

    scaler = load_scaler(exp_config["name"])
    feature_names = scaler.get_feature_names_out()

    class_names = ["churn", "no_churn"]

    for _, c in classifiers.items():
        clfs = load_cv_clfs(exp_config["name"], c["class_name"], n_splits)
        clf_visualizations.append(visualize_model(clfs, feature_names, class_names))
    
    return clf_visualizations















