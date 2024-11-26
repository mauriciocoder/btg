# In this script I train some binary models based on scikitlearn implementation of logistic regression.
# I have plotted the data distribution for test values and the model curve for the testing data.
# This is a nice script to check the implementation of a surface.
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


def plot_scatter(x_1: np.ndarray, x_2: np.ndarray, labels: np.ndarray) -> None:
    # Create the figure and axis
    fig, ax = plt.subplots()
    ax.scatter(x_1, x_2, c=labels, s=10, alpha=0.5)
    save_fig(fig, "scatter-classification-data-analysis")


def show_cost_curve(cost_values: list[float]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(range(0, len(cost_values)), cost_values, color="b")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    ax.set_title("Cost per iteration")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(
        "../../../data/classification-generation.csv",
        dtype={"x": "float32", "y": "float32", "z": "int8"},
    )
    x = df[["x", "y"]].values
    y = df["z"].values.reshape(df["z"].shape[0], 1)
    x = MinMaxScaler().fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {acc}")

    # Print the 3dplot scatter plot and the cost curve
    show_3d_scatter(
        x_test[:, 0].flatten(), x_test[:, 1].flatten(), y_predict.flatten(), model
    )
