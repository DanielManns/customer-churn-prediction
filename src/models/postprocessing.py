from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, BaseDecisionTree
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.models.preprocessing import get_exp_df, transform_df
from src.utility.plotting import plot_feature_importance
from src.utility.utility import load_exp_config, create_pp_dirs, load_clf


def run_postprocessing_session(exp_names: list[str]) -> None:
    """
    Starts postprocessing analysis of multiple experiments.

    :param exp_names: list[str] - list of experiment names
    :return: None
    """

    for exp_name in exp_names:
        create_pp_dirs(exp_name)
        run_postprocessing(exp_name)


def run_postprocessing(exp_name: str) -> None:
    """
    Starts postprocessing analysis of a single experiment.

    :param exp_name: str - experiment name
    :return: None
    """

    exp_config = load_exp_config(exp_name)
    clf_cons = exp_config["classifiers"]

    for _, clf_con in clf_cons.items():
        X, y = get_exp_df(clf_con["type"], exp_config["features"]["is_subset"])
        X, y, col_transformer = transform_df(X, y)
        clf = eval(clf_con["class_name"])()
        clf = load_clf(exp_name, clf)

        feature_names = col_transformer.get_feature_names_out()

        feature_importance = get_feature_importance(clf, feature_names)
        if isinstance(feature_importance, pd.DataFrame):
            plot_feature_importance(feature_importance, clf.__class__.__name__)


def get_feature_importance(clf: ClassifierMixin, feature_names: [str]) -> Optional[pd.DataFrame]:
    """
    Returns feature importance of already trained classifier (train_test_split).

    :param clf: sklearn.base.ClassifierMixin - Classifier
    :param feature_names: [str] - List of feature names
    :return: pd.DataFrame - feature importance
    """

    # coefficients in Logistic Regression describe conditional dependencies:
    # Example: y = wage, c_age = -0.4, c_experience = 0.4
    # Explanation:
    #   Increase in age will lead to a decrease in wage if all other features remain constant
    #   Increase in experience will lead to an increase in wage if all other features remain constant
    if isinstance(clf, LogisticRegression):
        im_data = clf.coef_.T

    elif isinstance(clf, BaseDecisionTree):
        im_data = clf.feature_importances_

    else:
        return None
        # raise ValueError("unexpected estimator")

    feature_importance = pd.DataFrame(
            im_data,
            columns=["Feature Importance"],
            index=feature_names
        )

    return feature_importance


def get_cv_feature_importance(clfs: [ClassifierMixin], feature_names: [str]) -> Optional[pd.DataFrame]:
    """
    Returns feature importance of already trained classifier (cross validation).

    :param clfs: [sklearn.base.ClassifierMixin] - List of classifiers from cross validation
    :param feature_names: [str] - List of feature names
    :return: pd.DataFrame - feature importance
    """

    if isinstance(clfs[0], LogisticRegression):
        feature_importance = pd.DataFrame(
            [
                clf.coef_ for clf in clfs["estimator"]
            ],
            columns=feature_names,
        )

    elif isinstance(clfs[0], BaseDecisionTree):
        feature_importance = pd.DataFrame(
            [
                clf.coef_ for clf in clfs
            ],
            columns=feature_names,
        )
    else:
        feature_importance = None
        # raise ValueError("unexpected estimator")

    return feature_importance


def get_permutation_importance(clf: ClassifierMixin, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """
    Returns permutation importance of given already trained pipeline and data.

    :param clf: sklearn.base.ClassifierMixin - Trained classifier
    :param X: pd.DataFrame - Data (train OR test)
    :param y: pd.DataFrame - Labels (train OR test)
    :return: pd.DataFrame - Permutation importance
    """

    result = permutation_importance(clf, X, y, n_repeats=10, random_state=12)
    sorted_importances_idx = result.importances_mean.argsort()
    perm_importance = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X.columns[sorted_importances_idx],
    )

    return perm_importance
