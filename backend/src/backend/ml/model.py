from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin, clone
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import BaseDecisionTree


def train_model(clf: BaseDecisionTree, X: pd.DataFrame, y: pd.DataFrame, cv_method, metric) -> \
        list[[ClassifierMixin], [float], [float]]:
    """
    Applies cross validation to given pipeline, data and labels. This includes training and evaluation.

    :param X: pd.DataFrame - data (train AND test data)
    :param y: pd.DataFrame - labels (train AND test labels)
    :param clf: sklearn.pipeline Pipeline - Pipeline
    :param cv_method: sklearn.model_selection._validation - cross validation method
    :return: [[ClassifierMixin], [float], [float]] - List of N classifiers, train_scores and test_scores
    """

    idxs = [i for i in cv_method.split(X, y)]

    cv_clfs = []
    train_scores = []
    test_scores = []

    for train_idx, test_idx in idxs:
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        if isinstance(clf, BaseDecisionTree):
            ccp_path = clf.cost_complexity_pruning_path(X_train, y_train)
            ccp_alphas, impurities = ccp_path.ccp_alphas, ccp_path.impurities
            alpha_clfs = []
            alpha_train_scores = []
            alpha_test_scores = []
            for ccp_alpha in ccp_alphas:
                new_clf = clone(clf)
                new_clf = new_clf.set_params(ccp_alpha=ccp_alpha)
                new_clf.fit(X_train, y_train)

                pred_train = new_clf.predict(X_train)
                pred_test = new_clf.predict(X_test)

                alpha_clfs.append(new_clf)
                alpha_train_scores.append(metric(pred_train, y_train))
                alpha_test_scores.append(metric(pred_test, y_test))

            best_idx = np.argmax(np.array(alpha_test_scores))
            clf = alpha_clfs[best_idx]
            train_score = alpha_train_scores[best_idx]
            test_score = alpha_test_scores[best_idx]
        else:
            clf.fit(X_train, y_train)
            pred_train = clf.predict(X_train)
            pred_test = clf.predict(X_test)
            train_score = metric(pred_train, y_train)
            test_score = metric(pred_test, y_test)

        cv_clfs.append(clf)
        train_scores.append(train_score)
        test_scores.append(test_score)

    return cv_clfs, train_scores, test_scores


def predict_model(clfs: list[ClassifierMixin], X: pd.DataFrame) -> list[float, float]:
    """
    Predicts churn from through cross validation trained classifiers for given df
    :param clfs: [sklearn.base.ClassifierMixIn] - List of sklearn classifiers from cross validation
    :param X: pd.Dataframe - Df to run inference on
    :return: [float, float] - mean and standard deviation of ensemble prediction
    """

    preds = np.array([clf.predict_exp(X) for clf in clfs])
    return preds.mean(axis=0), preds.std(axis=0)


def explain_model(clfs: list[ClassifierMixin], feature_names: list[str]) -> Optional[pd.DataFrame]:
    """
    Returns feature importance for a through cross validation obtained list of classifiers.

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
                clf.feature_importances_ for clf in clfs
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


