import matplotlib.pyplot as plt
from sklearn import tree
from skimage import io
import pandas as pd
from sklearn.tree import BaseDecisionTree
import plotly.express as px
from typing import List
import graphviz


def plot_feature_importance(df: pd.DataFrame, clf_idx):
    """
    Creates bar plot from given feature importance DataFrame
    :param df: pd.DataFrame - feature importance
    :param clf_idx: int - classifier index
    :return: fig - barplot figure
    """

    df["clf_idx"] = list(df.index)
    df = df.melt(id_vars=["clf_idx"], value_vars=list(df.columns), var_name="Feature", value_name="Importance")
    if isinstance(clf_idx, int):
        df = df.loc[df["clf_idx"] == clf_idx]
        df = df.sort_values(by="Importance")
        fig = px.bar(df, x="Importance", y="Feature", orientation="h", range_x=[0,1])
    else:
        mean = df.groupby("Feature")["Importance"].mean()
        std = df.groupby("Feature")["Importance"].std()
        df = pd.DataFrame({"Feature": list(mean.index), "Mean_Importance": list(mean), "Std": list(std)})
        df = df.sort_values(by="Mean_Importance")
        fig = px.bar(df, x="Mean_Importance", y="Feature", orientation="h", error_x="Std", range_x=[0,1])
    
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

def plot_dot_dt(dot_dts: List[str], clf_idx):
    g = graphviz.Source(dot_dts[clf_idx], format="jpeg")
    p = g.render("tree")
    img = io.imread(p)
    fig = px.imshow(img, binary_format="jpeg")
    return fig




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
