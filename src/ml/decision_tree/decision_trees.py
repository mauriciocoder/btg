# This exercise covers decision tree models from scikit-learn.
import numpy as np
from matplotlib import pyplot as plt
import os
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier, plot_tree


def save_fig(fig: plt.Figure, filename: str):
    output_dir = "../../../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"
    fig.savefig(os.path.join(output_dir, filename))


def plot_scatter(
    x: np.ndarray, y: np.ndarray, labels: np.ndarray
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x, y, c=labels)
    return fig, ax


def get_test_data(train_data: np.ndarray):
    x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
    y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))


def plot_color_predictions(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    predicted: np.ndarray,
):
    fig, axes = plt.subplots(figsize=(10, 10))
    axes.pcolormesh(test_data[0], test_data[1], predicted, cmap="autumn")
    axes.scatter(
        train_data[:, 0],
        train_data[:, 1],
        c=train_labels,
        s=100,
        cmap="autumn",
        edgecolors="black",
        linewidth=1.5,
    )
    save_fig(fig, "prediction_plot")


def plot_decision_tree(
    tree: DecisionTreeClassifier, feature_names: list[str], filename: str
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_tree(
        tree,
        filled=True,
        feature_names=feature_names,
        rounded=True,
        ax=ax,
    )
    save_fig(fig, "decision_tree_plot")
    return fig, ax


if __name__ == "__main__":
    train_data = np.concatenate(
        (np.random.normal(size=(100, 2)), np.random.normal(size=(100, 2), loc=2))
    )
    train_labels = np.concatenate((np.zeros(100), np.ones(100)))
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=17)
    tree.fit(train_data, train_labels)
    test_data = get_test_data(train_data)
    predicted = tree.predict(np.c_[test_data[0].ravel(), test_data[1].ravel()]).reshape(
        test_data[0].shape
    )
    plot_color_predictions(train_data, train_labels, test_data, predicted)
    plot_decision_tree(tree, ["x1", "x2"], "decision_tree_plot")
