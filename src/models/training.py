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
from sklearn.base import clone
from src.models.postprocessing import get_feature_importance
from src.models.preprocessing import get_con_features, get_cat_features, create_col_transformer, \
    apply_preprocessing
from src.config import config
import os

from src.utility.plotting import plot_feature_importance, plot_DT, plot_alpha_score_curve

con = config()
cv = True


def run_experiment_session(exp_names: list[str]) -> None:
    """
    Run multiple experiments from given list of experiment names.

    :param exp_names: list[str] - experiment names
    :return: None
    """

    for exp_name in exp_names:
        run_experiment(exp_name)


def run_experiment(exp_name: str) -> None:
    """
    Runs single experiment with given experiment name.

    :param exp_name: str - experiment name
    :return: None
    """

    exp_path = get_exp_path(exp_name)
    print(f"\nRunning experiment located at {exp_path} ...")

    exp_config = get_exp_config(exp_path)

    classifiers = exp_config["classifiers"]
    cv_method = eval(exp_config["cross_validation"]["class_name"])
    cv_method_params = exp_config["cross_validation"]["params"]
    test_ratio = exp_config["training"]["test_ratio"]
    train_seed = con.m_config.train_seed

    cat_X, con_X, mixed_X, y = get_exp_dfs(exp_config)

    for _, c in classifiers.items():
        c_class = eval(c["class_name"])
        c_params = c["params"]
        clf = c_class(**c_params)

        if c["type"] == "categorical":
            X = cat_X
        elif c["type"] == "continuous":
            X = con_X
        else:
            X = mixed_X

        # append training seed if classifiers has random component
        if "random_state" in clf.get_params().keys():
            clf.set_params(random_state=train_seed)

        # transform data
        col_transformer = create_col_transformer(X)
        X = col_transformer.fit_transform(X)

        if isinstance(clf, BaseDecisionTree):
            best, alphas, scores, train_scores = find_best_ccp_alpha(clf, X, y)
            clf.set_params(ccp_alpha=best[0])
            # plot_alpha_score_curve(train_scores, scores, alphas)

        if cv:
            # cross validate classifier
            clfs, scores = cross_validate_clf(clone(clf), X, y, cv_method(**cv_method_params))
            clf = clfs[0]
            score = scores.mean()
        else:
            # train classifier with train_test_split
            clf, train_score, score = tts_train_clf(clf, X, y, test_ratio=test_ratio)
            print(f"Train accuracy score of {clf.__class__.__name__}: {train_score}")

        print(f"Test accuracy score of {clf.__class__.__name__}: {score}")

        if clf.__class__.__name__ in ["DecisionTreeClassifier", "LogisticRegression"]:
            # feature_importance = get_feature_importance(pipe)
            # plot_feature_importance(feature_importance, classifier.__class__.__name__)
            if isinstance(clf, DecisionTreeClassifier):
                plot_DT(clf, feature_names=col_transformer.get_feature_names_out(), class_names=["No churn", "Churn"])
        print("\n")


def find_best_ccp_alpha(clf, X, y):
    # grow full tree with entire dataset, get ccp_path
    ccp_path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas, impurities = ccp_path.ccp_alphas, ccp_path.impurities
    best = (0.0, 0.0)
    alphas, scores, train_scores = [], [], []

    for ccp_alpha in ccp_alphas:
        new_clf = clone(clf)
        new_clf.set_params(ccp_alpha=ccp_alpha)

        # cv produces N clfs and scores
        cv_result = cross_validate(new_clf, X, y, cv=5, return_train_score=True)
        cv_test_scores = cv_result["test_score"]
        cv_train_scores = cv_result["train_score"]
        mean_score = cv_test_scores.mean()
        mean_train_score = cv_train_scores.mean()
        alphas.append(ccp_alpha)
        scores.append(mean_score)
        train_scores.append(mean_train_score)

        if mean_score > best[1]:
            best = (ccp_alpha, mean_score)
    return best, alphas, scores, train_scores


def find_best_ccp_clf(clfs: [BaseEstimator], scores) -> [BaseEstimator, float]:
    """
    Returns pipeline with the best test score from list of ccp pipelines.

    :param clfs: [Pipeline] - list of pipelines with decision tree classifier
    :param train_scores: [float] - list of train scores of each Pipeline
    :param scores: [float] - list of test scores of each Pipeline
    :return: Pipeline - Pipeline wit the highest test score
    """

    idx = np.argmax(np.array(scores))
    return clfs[idx], scores[idx]


def create_pipeline(col_transformer: ColumnTransformer, classifier: BaseEstimator) -> Pipeline:
    """
    Creates a sklearn pipeline from a ColumnTransformer and a classifier.

    :param col_transformer: sklearn.compose ColumnTransformer - ColumnTransformer
    :param classifier: sklearn.base.BaseEstimator - sklearn classifier
    :return: sklearn.pipeline Pipeline - Pipeline
    """

    return Pipeline(steps=[("col_transformer", col_transformer), ("classifier", classifier)])


def tts_train_clf(clf: BaseEstimator, X: pd.DataFrame, y: pd.DataFrame, test_ratio=0.2) -> [BaseEstimator, float, float]:
    """
    Trains a pipeline from given training data and training labels using a classical train-test split method.

    :param clf: sklearn.base.BaseEstimator - Classifier
    :param X: pd.DataFrame - training data
    :param y: pd.DataFrame - training labels
    :param test_ratio: float - ratio of train set size / test set size
    :param ccp: bool - returns ccp path if true

    :return: sklearn.base.BaseEstimator - Trained Classifier
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=con.m_config.train_seed)

    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    return clf, train_score, test_score


def cross_validate_clf(clf: BaseEstimator, X: pd.DataFrame, y: pd.DataFrame, cv_method: BaseCrossValidator) -> float:
    """
    Applies cross validation to given pipeline, data and labels. This includes training and evaluation.

    :param X: pd.DataFrame - data (train AND test data)
    :param y: pd.DataFrame - labels (train AND test labels)
    :param clf: sklearn.pipeline Pipeline - Pipeline
    :param cv_method: sklearn.model_selection._validation - cross validation method
    :return: float - mean cross validation score
    """
    cv_result = cross_validate(clf, X, y, cv=cv_method, return_estimator=True)
    scores = cv_result["test_score"]
    clfs = cv_result["estimator"]

    mean_score = scores.mean()
    std_score = scores.std()
    print(
        f"The {cv_method.__class__.__name__} score of {clf.__class__.__name__} is: "
        f"{mean_score:.3f} ± {std_score:.3f}"
    )

    return clfs, scores


def get_exp_dfs(exp_config: dict) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns preprocessed categorical-, continuous-, and mixed DataFrame as well as labels.

    :param exp_config: dict - eperimental configuration
    :return: list of pd.DataFrames - cat_X, con_X, mixed_X, y
    """

    raw_df = get_raw_data()
    mixed_df = apply_preprocessing(raw_df)

    y = mixed_df[con.m_config.target_name]
    mixed_df = mixed_df.drop(columns=[con.m_config.target_name])

    is_subset = exp_config["features"]["is_subset"]

    # subset important variables
    if is_subset:
        mixed_df = mixed_df.loc[:, con.m_config.im_vars]

    cat_df = mixed_df.drop(columns=get_con_features(mixed_df))
    con_df = mixed_df.drop(columns=get_cat_features(mixed_df))

    return  cat_df, con_df, mixed_df, y


def get_raw_data():
    """
    Returns raw DataFrame.

    :return: pd.DataFrame - raw data
    """

    return pd.read_csv(con.u_config.train_path)


def get_exp_path(exp_name):
    """
    Returns experiment path from given experiment name.

    :param exp_name: str- experiment name
    :return: str - experiment path
    """

    return os.path.join(con.u_config.exp_dir, exp_name)


def get_exp_config(exp_path):
    """
    Returns experimental config from given experiment path.

    :param exp_path: str - experiment path
    :return: dict - experiment config
    """

    with open(exp_path) as p:
        return yaml.safe_load(p)



