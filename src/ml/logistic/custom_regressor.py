# In this script I train some models based on my custom implementation of logistic regression.
# I have plotted the data distribution for test values and the model curve for the testing data.
# This is a nice script to check the implementation of a surface.
from dataclasses import dataclass
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils.plot import save_fig, show_3d_scatter
from scipy import optimize

# Required for live plot
# requires:sudo apt-get install python3-pyqt5
# pip install PyQt5
matplotlib.use("Qt5Agg")  # or 'Qt5Agg', 'MacOSX', etc.
# from utils.plot import save_fig, plot_linechart


@dataclass
class LogisticRegressor:
    """
    Model to fit and predict values using logistic regression.
    """

    theta: np.ndarray = None
    theta_trace: tuple[np.ndarray, float] = None  # (theta, cost) for each interaction

    @classmethod
    def _add_ones_column(cls, arr: np.ndarray) -> np.ndarray:
        ones_array = np.ones(shape=(arr.shape[0], 1), dtype=np.int8)
        return np.concatenate((ones_array, arr), axis=1, out=None)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_test = self._add_ones_column(x)
        return (self._sigmoid(x_test) > 0.5).astype(np.int8)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        # Implement sigmoid function
        h = np.dot(x, np.transpose(self.theta))
        return 1 / (1 + np.exp(-h))

    def _cost(self, theta: np.ndarray, *args: tuple[np.ndarray, np.ndarray]) -> float:
        # Cost function
        x = args[0]
        y = args[1]
        self.theta = np.transpose(theta.reshape(x.shape[1], 1))
        h = self._sigmoid(x)
        cost_func = -y * np.log(h) - (1 - y) * np.log(1 - h)
        cost = np.mean(cost_func)
        self.theta_trace.append((theta.copy(), cost))
        return cost

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.theta_trace = []
        x = self._add_ones_column(x_train)
        y = y_train
        # x_train.shape = (1822, 3)
        # y_train.shape = (1822, 1)
        initial_theta = np.zeros(x.shape[1], dtype=np.float64)
        optimize.minimize(fun=self._cost, x0=initial_theta, args=(x, y))


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
    model = LogisticRegressor()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    df_test = pd.DataFrame(
        {
            "x": x_test[:, 0].flatten(),
            "y": x_test[:, 1].flatten(),
            "z": y_test.flatten(),
        }
    )
    show_3d_scatter(
        df_test["x"].values, df_test["y"].values, df_test["z"].values, model
    )
    # Visually see how the loss decrease with each iteration
    cost_values = [t[1] for t in model.theta_trace]
    show_cost_curve(cost_values)
    """
    score = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {score}")
    """
