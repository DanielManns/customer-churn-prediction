import pandas as pd
import sklearn.base
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer


def create_pipeline(col_transformer: ColumnTransformer, classifier: sklearn.base.BaseEstimator) -> Pipeline:
    """
    Creates a sklearn pipeline from a ColumnTransformer and a classifier.

    :param col_transformer: sklearn.compose ColumnTransformer - ColumnTransformer
    :param classifier: sklearn.base.BaseEstimator - sklearn classifier
    :return: sklearn.pipeline Pipeline - Pipeline
    """

    return Pipeline(steps=[("col_transformer", col_transformer), ("classifier", classifier)])


def train_pipeline(pipe: Pipeline, X: pd.DataFrame, y: pd.DataFrame, test_ratio=0.2) -> sklearn.pipeline.Pipeline:
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

    #print(get_feature_importance(pipe, X_test, y_test))

    return pipe


def cv_train_pipeline(pipe: Pipeline, X: pd.DataFrame, y: pd.DataFrame, cv_method: sklearn.model_selection._validation, folds: int) -> float:
    """
    Applies cross validation to given pipeline, test_data and test_labels.

    :param X: pd.DataFrame - data (train AND test data)
    :param y: pd.DataFrame - labels (train AND test labels)
    :param pipe: sklearn.pipeline Pipeline - Pipeline
    :param cv_method: sklearn.model_selection._validation - cross validation method
    :param folds: int - number of cross validation folds
    :return: float - mean cross validation score
    """
    cv_result = cv_method(pipe, X, y, cv=folds, return_estimator=True)
    scores = cv_result["test_score"]
    pipelines = cv_result["estimator"]
    mean_score = scores.mean()
    print(
        f"The {cv_method.__name__} score of {pipe[1].__class__.__name__} is: "
        f"{mean_score:.3f} ± {scores.std():.3f}"
    )

    return pipelines


def get_feature_importance(pipe: sklearn.pipeline.Pipeline, X_test, y_test) -> dict:
    if isinstance(pipe[:-1], sklearn.naive_bayes.BaseEstimator):
        result = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=12)
        sorted_importances_idx = result.importances_mean.argsort()
        importances = pd.DataFrame(
            result.importances[sorted_importances_idx].T,
            columns=X_test.columns[sorted_importances_idx],
        )
        print(importances.head())
    return importances



