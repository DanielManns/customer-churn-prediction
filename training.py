import pandas as pd
import sklearn.base
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def create_pipeline(col_transformer: ColumnTransformer, classifier: sklearn.base.BaseEstimator) -> Pipeline:
    """
    Creates a sklearn pipeline from a ColumnTransformer and a classifier.

    :param col_transformer: sklearn.compose ColumnTransformer - ColumnTransformer
    :param classifier: sklearn.base.BaseEstimator - sklearn classifier
    :return: sklearn.pipeline Pipeline - Pipeline
    """

    return Pipeline(steps=[("col_transformer", col_transformer), ("classifier", classifier)])


def train_pipeline(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Pipeline:
    """
    Trains a pipeline from given training data (X_train) and training labels (y_train).

    :param X_train: pd.DataFrame - training data
    :param y_train: pd.DataFrame - training labels
    :param pipe: sklearn.pipeline Pipeline - Pipeline
    :return: sklearn.pipeline Pipeline - Pipeline
    """

    pipe.fit(X_train, y_train)
    return pipe


def cross_validate_pipeline( pipe: Pipeline, X: pd.DataFrame, y: pd.DataFrame, cv_method: sklearn.model_selection._validation, folds: int) -> float:
    """
    Applies cross validation to given pipeline, test_data and test_labels.

    :param X: pd.DataFrame - data (train AND test data)
    :param y: pd.DataFrame - labels (train AND test labels)
    :param pipe: sklearn.pipeline Pipeline - Pipeline
    :param cv_method: sklearn.model_selection._validation - cross validation method
    :param folds: int - number of cross validation folds
    :return: float - mean cross validation score
    """
    cv_result = cv_method(pipe, X, y, cv=folds)
    scores = cv_result["test_score"]
    mean_score = scores.mean()
    print(
        f"The {cv_method.__name__} score of {pipe[1].__class__.__name__} is: "
        f"{mean_score:.3f} ± {scores.std():.3f}"
    )

    return mean_score



