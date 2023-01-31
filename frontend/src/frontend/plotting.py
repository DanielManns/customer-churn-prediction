import matplotlib.pyplot as plt
from sklearn import tree
# from IPython.display import Image, display
# display(Image(filename="causal_model.png"))
import pandas as pd
from sklearn.tree import BaseDecisionTree
import seaborn as sns


def plot_feature_importance(f_imp_df: pd.DataFrame, classifier_name: str) -> None:
    """
    Creates bar plot from given feature importance DataFrame
    :param feature_importance: pd.DataFrame - feature importance
    :param classifier_name: str - name of classifier for caption
    :return: None
    """

    ax = f_imp_df.plot.barh(figsize=(15, 10))
    plt.title(classifier_name)
    plt.axvline(x=0, color=".5")
    ax.set_xlabel("Feature importance")
    ax.yaxis.label.set_size("x-small")

    plt.subplots_adjust(left=0.3)
    plt.show()
    return 

def sns_plot_feature_importance(f_imp_df: pd.DataFrame):
    fig, ax = plt.subplots()
    print(f_imp_df.head())
    sns.barplot(data=f_imp_df, ax=ax)
    return fig


def plot_dt(dt: BaseDecisionTree, feature_names: list[str], class_names: list[str]) -> None:
    """
    Plots visualization of given decision tree.

    :param dt: sklearn.tree.BaseDecisionTree - Decision Tree
    :param feature_names: [str] - list of feature names
    :param class_names: [str] - list of class names
    :return: None
    """

    fig, ax = plt.subplots()
    tree.plot_tree(dt, feature_names=feature_names, class_names=class_names, ax=ax, filled=True, proportion=True)
    plt.show()


def plot_alpha_score_curve(train_scores: list[float], test_scores: list[float], ccp_alphas: list[float]) -> None:
    """
    Plots the train- vs. test accuracy curve of different alphas.

    :param train_scores: [float] - train scores
    :param test_scores: [float] - test scores
    :param ccp_alphas: [float] - alpha values
    :return: None
    """

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs. alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
