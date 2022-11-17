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
    tree.plot_tree(dt, feature_names=feature_names, class_names=["No churn", "Churn"])
    plt.show()