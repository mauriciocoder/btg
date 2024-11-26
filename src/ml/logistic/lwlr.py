# This script trains a locally weighted logistic regression model.
# For each test point to be predicted, weights are generated for every training point (based on TAU hyperparameter).
# Then a prediction is made. This process is called a lazy learner.
# Notice how this version provides a better accuracy compared to non weighted logistic regression.
from dataclasses import dataclass
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils.plot import save_fig, show_3d_scatter
from scipy import optimize

# Required for live plot
# requires:sudo apt-get install python3-pyqt5
# pip install PyQt5
matplotlib.use("Qt5Agg")  # or 'Qt5Agg', 'MacOSX', etc.
# from utils.plot import save_fig, plot_linechart


def load_weights(
    x_train: np.ndarray, x_test_i: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    # x_train.shape = (m, n), where m = number of training examples, n = number of features
    # x_test_i.shape = (1, n)
    dists = np.power(x_test_i - x_train, 2).sum(axis=1)
    # dists.shape = (m, 1), each point has a distance to x_test_i
    # Weights are based on Gaussian formula
    # weights.shape = (m, 1). each point has a corresponding weight
    return np.exp(-np.power(dists, 2) / (2 * np.power(tau, 2)))


def show_scatter(x_1: np.ndarray, x_2: np.ndarray, labels: np.ndarray) -> None:
    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.scatter(x_1, x_2, c=labels, s=10, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(
        "../../../data/classification-generation-lwr.csv",
        dtype={"x": "float32", "y": "float32", "z": "int8"},
    )
    x = df[["x", "y"]].values
    y = df["z"].values.reshape(df["z"].shape[0], 1)
    x = MinMaxScaler().fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    show_scatter(x_train[:, 0], x_train[:, 1], y_train)

    y_predict = []
    for x_test_i in x_test:
        weights = load_weights(x_train, x_test_i, 0.0005)
        model = LogisticRegression()
        result = model.fit(x_train, y_train.flatten(), sample_weight=weights).predict(
            [x_test_i]
        )
        y_predict.append(result)

    acc = accuracy_score(y_test, y_predict)
    print(f"Accuracy for LWR: {acc}")

    print("Start to train a non weighted logistic Regression model")
    model = LogisticRegression()
    model.fit(x_train, y_train.flatten())
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print(f"Accuracy for non weighted Logistic Regression model: {acc}")
