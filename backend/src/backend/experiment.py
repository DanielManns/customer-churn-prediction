import pandas as pd
from sklearn.model_selection import cross_validate

import numpy as np
from sklearn.tree import BaseDecisionTree
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import clone, ClassifierMixin
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from preprocessing import get_preprocessed_dataset, scale_df
from config import BackendConfig
from utility import load_cv_clfs, save_clfs, save_scaler, load_scaler

conf = BackendConfig.from_yaml()


def start_training(exp_config: dict) -> [[ClassifierMixin], [float], [float]]:
    """
    Runs training for given experimental configuration.

    :param exp_config: dict - experiment configuration
    :return:
    """

    X, y = get_preprocessed_dataset(exp_config, train=True)
    X, y, scaler = scale_df(X, y)
    save_scaler(exp_config["name"], scaler)

    train_results = train_clfs(exp_config, X, y)

    return train_results


def train_clfs(exp_config: dict, X: pd.DataFrame, y: pd.DataFrame):
    """
    Trains AND saves all classifiers in given experiment configuration.

    :param exp_config: dict - experiment configuration
    :param X: pd.DataFrame - train data
    :param y: pd.DataFrame - labels
    :return: [dict] - list of training results
    """
    classifiers = exp_config["classifiers"]
    cv_method = eval(exp_config["cross_validation"]["class_name"])(**exp_config["cross_validation"]["params"])

    result = []

    for _, c in classifiers.items():
        clf = eval(c["class_name"])(**c["params"])

        if isinstance(clf, BaseDecisionTree):
            best, alphas, train_scores, test_scores = find_best_ccp_alpha(clf, X, y)
            clf.set_params(ccp_alpha=best[0])

        # cross validate classifier
        clfs, train_scores, test_scores = cross_validate_clf(clone(clf), X, y, cv_method)
        save_clfs(exp_config["name"], clfs)

        mean_train_score = np.round(train_scores.mean(), 3)
        mean_test_score = np.round(test_scores.mean(), 3)

        std_train_score = np.round(train_scores.std(), 3)
        std_test_score = np.round(test_scores.std(), 3)

        print(f"Train accuracy score of {clfs[0].__class__.__name__}: {mean_train_score} ± {std_train_score}")
        print(f"Test accuracy score of {clfs[0].__class__.__name__}: {mean_test_score} ± {std_test_score}")

        tmp = {
            "class_name": c["class_name"],
            "clfs": clfs,
            "mean_train_score": mean_train_score,
            "mean_test_score": mean_test_score,
            "std_train_score": std_train_score,
            "std_test_score": std_test_score
        }
        result.append(tmp)

    return result


def predict(exp_config: dict, X: pd.DataFrame) -> pd.DataFrame:
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

        preds = np.array([clf.predict_exp(X) for clf in clfs])
        mean_clf_preds.append(preds.mean(axis=0))
        std_clf_preds.append(preds.std(axis=0))

    mean_clf_preds = np.array(mean_clf_preds).T
    clf_names = [clf_name + "_mean_pred" for clf_name in list(classifiers.keys())]
    df = pd.DataFrame(data=mean_clf_preds, index=np.arange(X.shape[0]), columns=clf_names)

    return df


def find_best_ccp_alpha(clf: BaseDecisionTree, X: pd.DataFrame, y: pd.DataFrame) -> [(float, float), [float], [float], [float]]:
    """
    Finds the best cross validated alpha value for a single DecisionTree.

    :param clf: sklearn.tree.DecisionTreeClassifier - Untrained but initialised classifier.
    :param X: pd.DataFrame - Data (train and test)
    :param y: pd.DataFrame - Labels (train and test)
    :return: [(float, float), [float], [float], [float]] - best_alpha, all_alphas, all_mean_train_scores, all_mean_test_scores
    """

    # grow full tree with entire dataset, get ccp_path
    ccp_path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas, impurities = ccp_path.ccp_alphas, ccp_path.impurities
    best = (0.0, 0.0)
    alphas, test_scores, train_scores = [], [], []

    for ccp_alpha in ccp_alphas:
        new_clf = clone(clf)
        new_clf.set_params(ccp_alpha=ccp_alpha)

        # cv produces N clfs and scores
        cv_result = cross_validate(new_clf, X, y, cv=5, return_train_score=True)
        cv_test_scores = cv_result["test_score"]
        cv_train_scores = cv_result["train_score"]
        mean_test_score = cv_test_scores.mean()
        mean_train_score = cv_train_scores.mean()

        alphas.append(ccp_alpha)
        train_scores.append(mean_train_score)
        test_scores.append(mean_test_score)

        if mean_test_score > best[1]:
            best = (ccp_alpha, mean_test_score)
    return best, alphas, train_scores, test_scores


def find_best_ccp_clf(clfs: [ClassifierMixin], test_scores) -> [ClassifierMixin, float]:
    """
    Returns pipeline with the best test score from list of ccp pipelines.

    :param clfs: [Pipeline] - list of pipelines with decision tree classifier
    :param test_scores: [float] - list of test scores of each Pipeline
    :return: Pipeline - Pipeline wit the highest test score
    """

    idx = np.argmax(np.array(test_scores))
    return clfs[idx], test_scores[idx]


def cross_validate_clf(clf: ClassifierMixin, X: pd.DataFrame, y: pd.DataFrame, cv_method: BaseCrossValidator) -> \
        [[ClassifierMixin], [float], [float]]:
    """
    Applies cross validation to given pipeline, data and labels. This includes training and evaluation.

    :param X: pd.DataFrame - data (train AND test data)
    :param y: pd.DataFrame - labels (train AND test labels)
    :param clf: sklearn.pipeline Pipeline - Pipeline
    :param cv_method: sklearn.model_selection._validation - cross validation method
    :return: [[ClassifierMixin], [float], [float]] - List of N classifiers, train_scores and test_scores
    """

    cv_result = cross_validate(clf, X, y, cv=cv_method, return_estimator=True, return_train_score=True)
    train_scores = cv_result["train_score"]
    test_scores = cv_result["test_score"]
    clfs = cv_result["estimator"]

    return clfs, train_scores, test_scores












