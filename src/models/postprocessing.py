from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.tree import DecisionTreeClassifier, BaseDecisionTree
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.utils import Bunch







def get_feature_importance(pipe: Pipeline) -> pd.DataFrame:
    """
    Returns feature importance of already trained pipeline (train_test_split).

    :param pipe: sklearn.pipeline Pipeline - Pipeline
    :return: pd.DataFrame - feature importance
    """

    feature_names = pipe[:-1].get_feature_names_out()

    # coefficients in Logistic Regression describe conditional dependencies:
    # Example: y = wage, c_age = -0.4, c_experience = 0.4
    # Explanation:
    #   Increase in age will lead to a decrease in wage if all other features remain constant
    #   Increase in experience will lead to an increase in wage if all other features remain constant
    if isinstance(pipe[-1], LogisticRegression):
        im_data = pipe[-1].coef_.T

    elif isinstance(pipe[-1], BaseDecisionTree):
        im_data = pipe[-1].feature_importances_

    else:
        raise ValueError("unexpected estimator")

    feature_importance = pd.DataFrame(
            im_data,
            columns=["Feature Importance"],
            index=feature_names
        )

    return feature_importance


def get_cv_feature_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.DataFrame, cv_method: BaseCrossValidator) -> pd.DataFrame:
    """
    Returns cross validated feature importance of given already trained Pipeline and data.

    :param pipe: sklearn.pipeline Pipeline - Pipeline
    :param X: pd.DataFrame - train and test data
    :param y: pd.DataFrame - train and test labels
    :param cv_method: sklearn.model_selection.BaseCrossValidator - cross validation method
    :return: pd.DataFrame - feature importance
    """
    feature_names = pipe[:-1].get_feature_names_out()
    cv_result = cross_validate(pipe, X, y, cv=cv_method, return_estimator=True, n_jobs=2)

    if isinstance(pipe[-1], LogisticRegression):
        feature_importance = pd.DataFrame(
            [
                est[-1].coef_ for est in cv_result["estimator"]
            ],
            columns=feature_names,
        )

    elif isinstance(pipe[-1], BaseDecisionTree):
        feature_importance = pd.DataFrame(
            [
                est[-1].coef_ for est in cv_result["estimator"]
            ],
            columns=feature_names,
        )
    else:
        raise ValueError("unexpected estimator")

    return feature_importance


def get_permutation_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """
    Returns permutation importance of given already trained pipeline and data.

    :param pipe: sklearn.pipeline Pipeline - Trained pipeline
    :param X: pd.DataFrame - Data (train OR test)
    :param y: pd.DataFrame - Labels (train OR test)
    :return: pd.DataFrame - Permutation importance
    """

    result = permutation_importance(pipe, X, y, n_repeats=10, random_state=12)
    sorted_importances_idx = result.importances_mean.argsort()
    perm_importance = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X.columns[sorted_importances_idx],
    )

    return perm_importance

