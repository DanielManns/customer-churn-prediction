import matplotlib.pyplot as plt


def plot_feature_importance(feature_importance):
    feature_importance.plot.barh(figsize=(9, 7))
    plt.title("Ridge model, small regularization")
    plt.axvline(x=0, color=".5")
    plt.xlabel("Raw coefficient values")
    plt.subplots_adjust(left=0.3)
    plt.show()