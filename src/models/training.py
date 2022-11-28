import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import BaseCrossValidator, RepeatedKFold
from sklearn.utils import Bunch
from sklearn.base import clone, ClassifierMixin

from src.models.postprocessing import get_feature_importance
from src.models.preprocessing import get_con_features, get_cat_features, create_col_transformer, \
    apply_preprocessing, get_exp_df, transform_df
from src.config import config
import os

from src.utility.plotting import plot_feature_importance, plot_dt, plot_alpha_score_curve
from src.utility.utility import get_exp_conf_path, load_exp_config, load_raw_data, create_exp_dirs, save_clf

con = config()
cv = True


def run_experiment_session(exp_names: list[str]) -> None:
    """
    Run multiple experiments from given list of experiment names.

    :param exp_names: list[str] - experiment names
    :return: None
    """

    for exp_name in exp_names:
        create_exp_dirs(exp_name)
        experiment_scores = []
        for i in range(con.m_config.iterations):
            experiment_scores.append(run_experiment(exp_name, i))
        print(np.array(experiment_scores).mean(axis=0))
        print(np.array(experiment_scores).std(axis=0))


def run_experiment(exp_name: str, i: int) -> [float]:
    """
    Runs single experiment with given experiment name.

    :param exp_name: str - experiment name
    :param i: int - iteration of the same experiment
    :return: None
    """

    exp_config = load_exp_config(exp_name)

    classifiers = exp_config["classifiers"]
    cv_method = eval(exp_config["cross_validation"]["class_name"])
    cv_method_params = exp_config["cross_validation"]["params"]
    test_ratio = exp_config["training"]["test_ratio"]

    clf_scores = []
    for _, c in classifiers.items():
        X, y = get_exp_df(c["type"], exp_config["features"]["is_subset"])
        X, y, col_transformer = transform_df(X, y)

        clf = eval(c["class_name"])(**c["params"])

        if isinstance(clf, BaseDecisionTree):
            best, alphas, train_scores, test_scores = find_best_ccp_alpha(clf, X, y)
            clf.set_params(ccp_alpha=best[0])
            # plot_alpha_score_curve(train_scores, test_scores, alphas)

        if cv:
            # cross validate classifier
            clfs, train_scores, test_scores = cross_validate_clf(clone(clf), X, y, cv_method(**cv_method_params))

            # choose first classifier arbitrarily from N folds
            clf = clfs[0]

            train_score = np.round(train_scores.mean(), 3)
            test_score = np.round(test_scores.mean(), 3)
        else:
            # train classifier with train_test_split
            clf, train_score, test_score = tts_train_clf(clf, X, y, test_ratio=test_ratio)
            train_scores, test_scores = train_score, test_score

        print(f"Train accuracy score of {clf.__class__.__name__}: {train_score} ± {np.round(train_scores.std(), 3)}")
        print(f"Test accuracy score of {clf.__class__.__name__}: {test_score} ± {np.round(test_scores.std(), 3)}")

        if clf.__class__.__name__ in ["DecisionTreeClassifier", "LogisticRegression"]:
            # feature_importance = get_feature_importance(clf, feature_names)
            # plot_feature_importance(feature_importance, clf.__class__.__name__)
            if isinstance(clf, DecisionTreeClassifier):
                pass
                # plot_DT(clf, feature_names=col_transformer.get_feature_names_out(), class_names=["No churn", "Churn"])

        save_clf(exp_name, clf, i)
        clf_scores.append(test_score)
        print("\n")
    return clf_scores


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


def create_pipeline(col_transformer: ColumnTransformer, classifier: ClassifierMixin) -> Pipeline:
    """
    Creates a sklearn pipeline from a ColumnTransformer and a classifier.

    :param col_transformer: sklearn.compose ColumnTransformer - ColumnTransformer
    :param classifier: sklearn.base.BaseEstimator - sklearn classifier
    :return: sklearn.pipeline Pipeline - Pipeline
    """

    return Pipeline(steps=[("col_transformer", col_transformer), ("classifier", classifier)])


def tts_train_clf(clf: ClassifierMixin, X: pd.DataFrame, y: pd.DataFrame, test_ratio=0.2) -> [ClassifierMixin, float, float]:
    """
    Trains a pipeline from given training data and training labels using a classical train-test split method.

    :param clf: sklearn.base.BaseEstimator - Classifier
    :param X: pd.DataFrame - training data
    :param y: pd.DataFrame - training labels
    :param test_ratio: float - ratio of train set size / test set size

    :return: [sklearn.base.ClassifierMixin], float, float] - Trained Classifier, train_score, test_score
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=con.m_config.train_seed)

    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    return clf, train_score, test_score


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












