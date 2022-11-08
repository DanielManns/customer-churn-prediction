import pandas as pd
import sklearn.base
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BaseEstimator as BaseEstimatorNB
from sklearn.tree import DecisionTreeClassifier, BaseDecisionTree
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedKFold, BaseCrossValidator


def create_pipeline(col_transformer: ColumnTransformer, classifier: BaseEstimator) -> Pipeline:
    """
    Creates a sklearn pipeline from a ColumnTransformer and a classifier.

    :param col_transformer: sklearn.compose ColumnTransformer - ColumnTransformer
    :param classifier: sklearn.base.BaseEstimator - sklearn classifier
    :return: sklearn.pipeline Pipeline - Pipeline
    """

    return Pipeline(steps=[("col_transformer", col_transformer), ("classifier", classifier)])


def train_pipeline(pipe: Pipeline, X: pd.DataFrame, y: pd.DataFrame, test_ratio=0.2) -> Pipeline:
    """
    Trains a pipeline from given training data and training labels using a classical train-test split method.

    :param pipe: sklearn.pipeline Pipeline - Pipeline
    :param X: pd.DataFrame - training data
    :param y: pd.DataFrame - training labels
    :param test_ratio: float - ratio of train set size / test set size

    :return: sklearn.pipeline Pipeline - Pipeline
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)

    pipe.fit(X_train, y_train)
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    print(f"Train accuracy score of {pipe['classifier'].__class__.__name__}: {train_score}")
    print(f"Test accuracy score of {pipe['classifier'].__class__.__name__}: {test_score}")

    # print(get_feature_importance(pipe, X_test, y_test))

    return pipe


def cv_pipeline(pipe: Pipeline, X: pd.DataFrame, y: pd.DataFrame, cv_method: BaseCrossValidator) -> float:
    """
    Applies cross validation to given pipeline, data and labels. This includes training and evaluation.

    :param X: pd.DataFrame - data (train AND test data)
    :param y: pd.DataFrame - labels (train AND test labels)
    :param pipe: sklearn.pipeline Pipeline - Pipeline
    :param cv_method: sklearn.model_selection._validation - cross validation method
    :param folds: int - number of cross validation folds
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



