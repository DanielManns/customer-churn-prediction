import pandas as pd
import yaml
import gradio as gr
import joblib
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
from functools import partial

from src.models.postprocessing import get_feature_importance
from src.models.preprocessing import get_con_features, get_cat_features, create_col_transformer, \
    apply_preprocessing, get_preprocessed_dataset, scale_df
from src.config import config
import os

from src.utility.plotting import plot_feature_importance, plot_dt, plot_alpha_score_curve
from src.utility.utility import get_exp_conf_path, load_exp_config, load_train_dataset, create_exp_dirs, save_clf, \
    load_clf, \
    load_clfs, load_test_dataset, save_clfs, save_scaler, load_scaler
from IPython.display import display

con = config()


def run_training(exp_config: dict) -> [[ClassifierMixin], [float], [float]]:
    """
    Runs training for given experimental configuration.

    :param exp_config: dict - experiment configuration
    :return:
    """

    classifiers = exp_config["classifiers"]
    features = exp_config["features"]["is_subset"]
    cv_method = eval(exp_config["cross_validation"]["class_name"])(**exp_config["cross_validation"]["params"])

    for _, c in classifiers.items():
        X, y = get_preprocessed_dataset(c["type"], features, mode="train")
        X, y, scaler = scale_df(X, y)

        clf = eval(c["class_name"])(**c["params"])

        if isinstance(clf, BaseDecisionTree):
            best, alphas, train_scores, test_scores = find_best_ccp_alpha(clf, X, y)
            clf.set_params(ccp_alpha=best[0])
            # plot_alpha_score_curve(train_scores, test_scores, alphas)

        # cross validate classifier
        clfs, train_scores, test_scores = cross_validate_clf(clone(clf), X, y, cv_method)
        save_clfs(exp_config["name"], clfs)
        save_scaler(exp_config["name"], scaler, c["class_name"])

        mean_train_score = np.round(train_scores.mean(), 3)
        mean_test_score = np.round(test_scores.mean(), 3)

        std_train_score = np.round(train_scores.std(), 3)
        std_test_score = np.round(test_scores.std(), 3)

        print(f"Train accuracy score of {clfs[0].__class__.__name__}: {mean_train_score} ± {std_train_score}")
        print(f"Test accuracy score of {clfs[0].__class__.__name__}: {mean_test_score} ± {std_test_score}")
        print("\n")

    return


def run_inference(exp_config: dict, X: pd.DataFrame = None) -> pd.DataFrame:
    """
    Runs inference for given experiment configuration.

    :param exp_config: dict - experiment configuration
    :param X: pd.DataFrame - Data to run inference on.
    :return: pd.DataFrame - Predictions
    """

    classifiers = exp_config["classifiers"]
    n_splits = exp_config["cross_validation"]["params"]["n_splits"]
    mean_clf_preds = []
    std_clf_preds = []

    for _, c in classifiers.items():
        clfs = load_clfs(exp_config["name"], c["class_name"], n_splits)
        scaler = load_scaler(exp_config["name"], c["class_name"])
        X, _ = get_preprocessed_dataset(c["type"], exp_config["features"]["is_subset"], mode="test")
        X = scaler.transform(X)

        preds = np.array([clf.predict(X) for clf in clfs])
        mean_clf_preds.append(preds.mean(axis=0))
        #std_clf_preds.append(preds.std(axis=0))
    # mean_clf_preds.shape = (num_clfs, num_preds)
    mean_clf_preds = np.array(mean_clf_preds).T
    # std_clf_preds = np.array(std_clf_preds).T
    # data = np.concatenate((mean_clf_preds, std_clf_preds), axis=1)
    data = mean_clf_preds
    c_names_1 = [clf_name + "_mean_pred" for clf_name in list(classifiers.keys())]
    #c_names_2 = [clf_name + "_std_pred" for clf_name in list(classifiers.keys())]
    #c_names = c_names_1 + c_names_2
    df = pd.DataFrame(data=data, index=np.arange(X.shape[0]), columns=c_names_1)

    return df


def run_gui(exp_config):
    df, _ = get_preprocessed_dataset("mixed", exp_config["features"]["is_subset"], mode="test")
    inputs = [gr.Dataframe(row_count=(1, "dynamic"), col_count=(26, "fixed"), label="Input Data", interactive=True)]

    outputs = [
        gr.Dataframe(row_count=(1, "dynamic"), col_count=(2, "fixed"), label="Predictions", headers=["Logistic Regression Churn Probability", "DT Churn Probability"])]

    # LR_clfs = load_clfs(exp_config["name"], "LogisticRegression", exp_config["cross_validation"]["n_splits"])
    # DT_clfs = load_clfs(exp_config["name"], "DecisionTreeClassifier", exp_config["cross_validation"]["n_splits"])

    gr.Interface(fn=partial(run_inference, exp_config), inputs=inputs, outputs=outputs, examples=[[df.head(50)]], examples_per_page=10).launch()


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












