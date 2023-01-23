import pandas as pd
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from backend.ml.model import train_model, predict_model, explain_model
from backend.ml.preprocessing import get_train_dataset, scale_df
from backend.config import conf
from backend.ml.utility import load_cv_clfs, save_clfs, save_scaler, load_scaler

CV_METHOD = RepeatedKFold
METRIC = accuracy_score

def train_experiment(exp_config: dict) -> list[list[ClassifierMixin], list[float], list[float]]:
    """
    Runs training for given experimental configuration.

    :param exp_config: dict - experiment configuration
    :return:
    """

    X, y = get_train_dataset(exp_config)
    X, y, scaler = scale_df(X, y)
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

        print()

    return result


def predict_experiment(exp_config: dict, X: pd.DataFrame) -> pd.DataFrame:
    """
    Runs inference for given experiment configuration.

    :param exp_config: dict - experiment configuration
    :param X: pd.DataFrame - Data to run inference on.
    :return: pd.DataFrame - Predictions
    """

    scaler = load_scaler(exp_config["name"])
    X = scaler.transform(X)

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
    clf_names = [clf_name + "_mean_pred" for clf_name in list(classifiers.keys())]
    df = pd.DataFrame(data=mean_clf_preds, index=np.arange(X.shape[0]), columns=clf_names)

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















