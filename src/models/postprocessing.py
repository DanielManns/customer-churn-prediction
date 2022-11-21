from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.tree import DecisionTreeClassifier, BaseDecisionTree
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.utils import Bunch

from src.models.preprocessing import create_col_transformer, get_exp_df
from src.utility.plotting import plot_feature_importance
from src.utility.utility import load_exp_models, get_exp_config


def run_postprocessing_session(exp_names: list[str]) -> None:
    """
    Starts postprocessing analysis of multiple experiments.

    :param exp_names: list[str] - list of experiment names
    :return: None
    """

    for exp_name in exp_names:
        run_postprocessing(exp_name)


def run_postprocessing(exp_name: str) -> None:
    """
    Starts postprocessing analysis of a single experiment.

    :param exp_name: str - experiment name
    :return: None
    """

    clfs, Xs, y = load_exp_models(exp_name)

    for clf, X in zip(clfs, Xs):
        col_transformer = create_col_transformer(X)
        X = col_transformer.fit_transform(X)
        feature_names = col_transformer.get_feature_names_out()
        feature_importance = get_feature_importance(clf, feature_names)
        plot_feature_importance(feature_importance, feature_names)


def get_feature_importance(clf: ClassifierMixin, feature_names: [str]) -> pd.DataFrame:
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
        raise ValueError("unexpected estimator")

    feature_importance = pd.DataFrame(
            im_data,
            columns=["Feature Importance"],
            index=feature_names
        )

    return feature_importance


def get_cv_feature_importance(clfs: [ClassifierMixin], feature_names: [str]) -> pd.DataFrame:
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
        raise ValueError("unexpected estimator")

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

