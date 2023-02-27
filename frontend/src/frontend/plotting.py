import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
from sklearn.tree import BaseDecisionTree
import seaborn as sns
import plotly.express as px


def plot_feature_importance(df: pd.DataFrame, clf_idx):
    """
    Creates bar plot from given feature importance DataFrame
    :param df: pd.DataFrame - feature importance
    :param clf_idx: int - classifier index
    :return: fig - barplot figure
    """
    if clf_idx != "avg":
        df = df.loc[clf_idx, :]
        fig = px.bar(df, orientation="h")
    else:
        mean = df.mean(axis=0)
        sd = df.std(axis=0)
        fig = px.bar(x = df.columns, y = mean.to_numpy(), orientation="h", error_y=sd.to_numpy())
    
    #fig, ax = plt.subplots(figsize=(20, 20))
    #sns.barplot(data=f_imp_df, ax=ax, errorbar="sd", orient="h")
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
