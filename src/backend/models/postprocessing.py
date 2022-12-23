from sklearn.inspection import permutation_importance
from sklearn.tree import BaseDecisionTree
import pandas as pd
from sklearn.base import ClassifierMixin
from typing import Optional
from sklearn.linear_model import LogisticRegression
from src.backend.models.preprocessing import get_preprocessed_dataset, scale_df
from src.backend.utility.plotting import plot_feature_importance
from src.backend.utility.utility import load_exp_config, create_pp_dirs, load_clf


def run_postprocessing_session(exp_names: list[str], reps: int) -> None:
    """
    Starts postprocessing analysis of multiple experiments.

    :param exp_names: list[str] - list of experiment names
    :param reps: int - experiment iteration
    :return: None
    """

    for exp_name in exp_names:
        create_pp_dirs(exp_name)
        run_postprocessing(exp_name, reps)


def run_postprocessing(exp_name: str, reps: int) -> None:
    """
    Starts postprocessing analysis of a single experiment.

    :param exp_name: str - experiment name
    :param reps: int - total number of experiment repetitions
    :return: None
    """

    exp_config = load_exp_config(exp_name)
    clf_cons = exp_config["classifiers"]

    for _, clf_con in clf_cons.items():
        X, y = get_preprocessed_dataset(clf_con["type"], exp_config["features"]["is_subset"])
        X, y, col_transformer = scale_df(X, y)
        clf = eval(clf_con["class_name"])()
        feature_names = col_transformer.get_feature_names_out()

        if isinstance(clf, LogisticRegression) or isinstance(clf, BaseDecisionTree):
            rep_f_imps = None
            for i in range(reps):
                try:
                    clf = load_clf(exp_name, clf, i)
                except FileNotFoundError:
                    print(f"Could not load classifier. Consider execution in training mode of experiment '{exp_name}'.")
                    return
                f_imp = get_feature_importance(clf, feature_names)
                if rep_f_imps is not None:
                    rep_f_imps = f_imp
                else:
                    rep_f_imps = pd.concat((rep_f_imps, f_imp))
            plot_feature_importance(rep_f_imps, clf.__class__.__name__)
        else:
            continue


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
        raise ValueError("unexpected estimator")

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
