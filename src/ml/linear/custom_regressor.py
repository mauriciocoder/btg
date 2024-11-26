from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

# from utils.plot import save_fig, plot_linechart


class Method(Enum):
    BATCH: str = "batch"
    NORMAL_EQUATION: str = "normal_equation"


@dataclass
class LinearRegressor:
    """
    Model to fit and predict values using linear regression. The implementation supports two
    methods: batch and normal equation.
    """

    # Hyperparameters
    alpha: float = 0.01  # learning rate for gradient descent
    num_iterations: int = (
        1000  # Number of iterations for attempting convergence. Required for Batch
    )
    method: Method = (
        Method.BATCH
    )  # Method to fit the model, either BATCH or NORMAL_EQUATION
    theta: np.ndarray = None  # Model parameters, initialized to None

    @classmethod
    def _add_ones_column(cls, arr: np.ndarray) -> np.ndarray:
        # Including an extra column of 1s in arr
        ones_array = np.ones(shape=(arr.shape[0], 1), dtype=np.int8)
        return np.concatenate((ones_array, arr), axis=1, out=None)

    """
    # This plot only works with a single feature data set
    def _plot_theta_chart(self, index: int, x: np.ndarray, y: np.ndarray):
        fig = plot_linechart(
            pd.DataFrame({"x": x.flatten(), "y": y.flatten()}),
            "x",
            "y",
        )
        y_test = self.predict(x)
        plot_linechart(
            pd.DataFrame({"x": x.flatten(), "y": y_test.flatten()}),
            "x",
            "y",
            fig=fig,
            plot_title=f"Theta={self.theta}",
        )
        save_fig(fig, f"fitting_theta_{index}")
    """

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_test = self._add_ones_column(x)
        return np.dot(self.theta, np.transpose(x_test)).flatten()

    def cost_function_derivative(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = np.dot(self.theta, np.transpose(x)) - np.transpose(y)
        return np.dot(diff, x)

    def _update_theta(self, x: np.ndarray, y: np.ndarray):
        inc = self.cost_function_derivative(x, y)
        m = x.shape[0]
        # print(f"self.theta={self.theta}")
        self.theta = self.theta - (self.alpha * (1.0 / m) * inc)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_train = self._add_ones_column(x)
        if self.method == Method.BATCH:
            self.theta = np.zeros(shape=(1, x_train.shape[1]), dtype=np.int8)
            for i in range(self.num_iterations):
                self._update_theta(x_train, y)
        elif self.method == Method.NORMAL_EQUATION:
            # Normal equation to compute optimal parameters theta in linear regression:
            #     theta = (X.T @ X)^(-1) @ X.T @ y
            x_trans = np.transpose(x_train)
            self.theta = np.dot(
                np.linalg.inv(np.dot(x_trans, x_train)), np.dot(x_trans, y)
            )
            self.theta = self.theta.reshape(1, x_train.shape[1])
            # self.theta = np.transpose(self.theta)


if __name__ == "__main__":

    def load_train_data() -> pd.DataFrame:
        dtype_mapping = {"x": "float32", "y": "float32"}
        return pd.read_csv("../../../data/linear-generation3.csv", dtype=dtype_mapping)

    df = load_train_data()
    x = df["x"].values.reshape(df["x"].shape[0], 1)
    y = df["y"].values.reshape(df["y"].shape[0], 1)
    model = LinearRegressor(method=Method.BATCH)
    model.fit(x, y)
    print(model.theta)
    model = LinearRegressor(method=Method.NORMAL_EQUATION)
    model.fit(x, y)
    print(model.theta)
