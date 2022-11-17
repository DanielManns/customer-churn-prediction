import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
from IPython.display import Image, display
# display(Image(filename="causal_model.png"))


def plot_feature_importance(feature_importance, classifier_name):
    feature_importance.plot.barh(figsize=(15, 10))
    plt.title(classifier_name)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Feature importance")
    plt.subplots_adjust(left=0.3)
    plt.show()


def plot_DT(dt, feature_names):
    fig, ax = plt.subplots()
    tree.plot_tree(dt, feature_names=feature_names, class_names=["No churn", "Churn"], ax=ax)
    fig.show()


def plot_alpha_score_curve(train_scores, test_scores, ccp_alphas):
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
