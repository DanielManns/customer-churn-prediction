import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import BaseCrossValidator, RepeatedKFold

from src.models.postprocessing import get_feature_importance
from src.models.preprocessing import get_con_features, get_cat_features, create_col_transformer, \
    apply_preprocessing
from src.config import config
import os

from src.utility.plotting import plot_feature_importance, plot_DT

c = config()


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

    ccp = False
    exp_path = get_exp_path(exp_name)
    print(f"\nRunning experiment located at {exp_path} ...")

    exp_config = get_exp_config(exp_path)
    cat_X, con_X, mixed_X, y = get_exp_dfs(exp_config)

    classifiers = exp_config["classifiers"]
    cv_method = eval(exp_config["cross_validation"]["class_name"])
    cv_method_params = exp_config["cross_validation"]["params"]

    for _, cl in classifiers.items():
        c_params = cl["params"]
        # append training seed if classifiers has random component
        if cl["class_name"] in c.m_config.ran_classifiers:
            c_params = {**c_params, **{"random_state": c.m_config.train_seed}}

        if cl["type"] == "categorical":
            X = cat_X
        elif cl["type"] == "continuous":
            X = con_X
        else:
            X = mixed_X

        col_transformer = create_col_transformer(X)
        classifier = eval(cl["class_name"])(**c_params)
        if isinstance(classifier, BaseDecisionTree):
            ccp = True

        pipe = create_pipeline(col_transformer, classifier)
        pipe, train_score, test_score, ccp_path = train_pipeline(pipe, X, y, test_ratio=exp_config["training"]["test_ratio"], ccp=ccp)

        # do cost complexity pruning
        if ccp and ccp_path:
            ccp_alphas, impurities = ccp_path.ccp_alphas, ccp_path.impurities
            pipes, train_scores, test_scores = [], [], []
            for ccp_alpha in ccp_alphas:
                c_params = {**c_params, **{"ccp_alpha": ccp_alpha}}
                classifier = eval(cl["class_name"])(**c_params)
                pipe = create_pipeline(col_transformer, classifier)
                p, train_score, test_score, _ = train_pipeline(pipe, X, y, ccp=False)
                pipes.append(p)
                train_scores.append(train_score)
                test_scores.append(test_score)
            print(train_scores)
            print(test_scores)
            pipe = find_best_ccp_pipe(pipes, test_scores)
        cv_pipelines = cross_validate_pipeline(pipe, X, y, cv_method(**cv_method_params))
        if classifier.__class__.__name__ in ["DecisionTreeClassifier", "LogisticRegression"]:
            feature_importance = get_feature_importance(pipe)
            #plot_feature_importance(feature_importance, classifier.__class__.__name__)
            if isinstance(classifier, DecisionTreeClassifier):
                plot_DT(classifier, feature_names=pipe[:-1].get_feature_names_out())
        print("\n")


def find_best_ccp_pipe(pipes: [Pipeline], test_scores) -> Pipeline:
    return pipes[np.argmax(np.array(test_scores))]


def create_pipeline(col_transformer: ColumnTransformer, classifier: BaseEstimator) -> Pipeline:
    """
    Creates a sklearn pipeline from a ColumnTransformer and a classifier.

    :param col_transformer: sklearn.compose ColumnTransformer - ColumnTransformer
    :param classifier: sklearn.base.BaseEstimator - sklearn classifier
    :return: sklearn.pipeline Pipeline - Pipeline
    """

    return Pipeline(steps=[("col_transformer", col_transformer), ("classifier", classifier)])


def train_pipeline(pipe: Pipeline, X: pd.DataFrame, y: pd.DataFrame, test_ratio=0.2, ccp=False) -> [Pipeline, float, float]:
    """
    Trains a pipeline from given training data and training labels using a classical train-test split method.

    :param pipe: sklearn.pipeline Pipeline - Pipeline
    :param X: pd.DataFrame - training data
    :param y: pd.DataFrame - training labels
    :param test_ratio: float - ratio of train set size / test set size

    :return: sklearn.pipeline Pipeline - Pipeline
    """
    ccp_path = None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)

    pipe.fit(X_train, y_train)

    if ccp and isinstance(pipe[-1], BaseDecisionTree):
        ccp_path = pipe[-1].cost_complexity_pruning_path(pipe[:-1].transform(X_train), y_train)

    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    print(f"Train accuracy score of {pipe['classifier'].__class__.__name__}: {train_score}")
    print(f"Test accuracy score of {pipe['classifier'].__class__.__name__}: {test_score}")

    return pipe, train_score, test_score, ccp_path


def cross_validate_pipeline(pipe: Pipeline, X: pd.DataFrame, y: pd.DataFrame, cv_method: BaseCrossValidator) -> float:
    """
    Applies cross validation to given pipeline, data and labels. This includes training and evaluation.

    :param X: pd.DataFrame - data (train AND test data)
    :param y: pd.DataFrame - labels (train AND test labels)
    :param pipe: sklearn.pipeline Pipeline - Pipeline
    :param cv_method: sklearn.model_selection._validation - cross validation method
    :return: float - mean cross validation score
    """
    cv_result = cross_validate(pipe, X, y, cv=cv_method, return_estimator=True)
    scores = cv_result["test_score"]
    pipelines = cv_result["estimator"]
    mean_score = scores.mean()
    print(
        f"The {cv_method.__class__.__name__} score of {pipe[1].__class__.__name__} is: "
        f"{mean_score:.3f} ± {scores.std():.3f}"
    )

    return pipelines


def get_exp_dfs(exp_config: dict) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns preprocessed categorical-, continuous-, and mixed DataFrame as well as labels.

    :param exp_config: dict - eperimental configuration
    :return: list of pd.DataFrames - cat_X, con_X, mixed_X, y
    """

    raw_df = get_raw_data()
    mixed_df = apply_preprocessing(raw_df)

    y = mixed_df[c.m_config.target_name]
    mixed_df = mixed_df.drop(columns=[c.m_config.target_name])

    is_subset = exp_config["features"]["is_subset"]

    # subset important variables
    if is_subset:
        mixed_df = mixed_df.loc[:, c.m_config.im_vars]

    cat_df = mixed_df.drop(columns=get_con_features(mixed_df))
    con_df = mixed_df.drop(columns=get_cat_features(mixed_df))

    return  cat_df, con_df, mixed_df, y


def get_raw_data():
    """
    Returns raw DataFrame.

    :return: pd.DataFrame - raw data
    """

    return pd.read_csv(c.u_config.train_path)


def get_exp_path(exp_name):
    """
    Returns experiment path from given experiment name.

    :param exp_name: str- experiment name
    :return: str - experiment path
    """

    return os.path.join(c.u_config.exp_dir, exp_name)


def get_exp_config(exp_path):
    """
    Returns experimental config from given experiment path.

    :param exp_path: str - experiment path
    :return: dict - experiment config
    """

    with open(exp_path) as p:
        return yaml.safe_load(p)



