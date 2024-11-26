# This script implements the locally weighted regression (LWR).
# Some models can be trained based on data which is not parametric (there is not a single linear equation which models the data).
# For such cases we can train a linear model for some regions closer to the data point we are trying to predict.
# Just like splitting the training data in sets, and each set is used to train a linear model. This approcah can
# capture partial linearity of the model.
# For a comprehensive discussion on this topic, check:https://www.youtube.com/watch?v=1UXWf2AOzu0&list=PLgaemU2xZlTisMjcQUvU8UzOpK4QXcUq-&index=22
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
from utils.ml import print_coefficients

warnings.filterwarnings("ignore", category=UserWarning)


def load_weights(
    x_train: np.ndarray, x_test_i: np.ndarray, tau: float = 0.1
) -> np.ndarray:
    # Compute the distance from x_test_i to each point in x_train
    dists = np.power(x_test_i - x_train, 2).sum(axis=1)
    # Compute the weights based on gaussian formula
    return np.exp(-np.power(dists, 2) / (2 * np.power(tau, 2)))


def train_and_predict_lasso_model(
    feature_names: list[str],
    x_train_scaled: np.ndarray,
    y_train: np.ndarray,
    x_test_scaled: np.ndarray,
    y_test: np.ndarray,
    tau=5,
) -> np.ndarray:
    print("### Training a lasso model")
    model = LassoCV(cv=5, selection="random")
    y_predict_test = []
    for x_test_i in x_test_scaled:
        weights = load_weights(x_train_scaled, x_test_i, tau)
        model.fit(x_train_scaled, y_train, weights)
        y_predict_test.append(model.predict(x_test_i.reshape(1, -1)))
    mse = mean_squared_error(y_test, y_predict_test)
    print(f"Alpha: {model.alpha_}")
    print(f"Mean squared error (test): {mse:.3f}")
    r_scores = r2_score(y_test, y_predict_test)
    print(f"R^2 scores for tau={tau}:", r_scores)
    print_coefficients(model, feature_names)
    return y_predict_test


if __name__ == "__main__":
    df = pd.read_csv("../../../data/concrete.csv")
    print(df.head())
    print(df.info())

    x = df.drop(columns=["Strength"]).values
    y = df["Strength"].values
    x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    # The best tau found is 5
    y_predict_test = train_and_predict_lasso_model(
        list(df.columns), x_train_scaled, y_train, x_test_scaled, y_test, tau=5
    )

    df = pd.DataFrame({"y_test": y_test, "y_predict_test": y_predict_test})
    print(df)

"""
Results:
None
R^2 scores for tau=1: [0.6681386610837307, 0.6592113516592926, 0.6632885307931552, 0.6641745562304318, 0.6665568098423353, 0.66577179650957, 0.6599752229871247, 0.667136995280665, 0.6672686517864029, 0.663404889758805]
R^2 scores for tau=2: [0.7379046210923884, 0.7378207040641707, 0.7376125614977258, 0.7373597606090903, 0.7386384704513309, 0.7382360571759589, 0.7371617977055498, 0.7385085873052376, 0.7376393106220975, 0.738403118682031]
R^2 scores for tau=5: [0.767978524963205, 0.768020049712875, 0.7678863236304574, 0.7680011865938884, 0.768131920789821, 0.7681345785893952, 0.7678731214497978, 0.7680987703873192, 0.7681090187852482, 0.7680746861266083]
R^2 scores for tau=10: [0.7596954092069572, 0.7597783891637289, 0.7596286814164139, 0.7597000564953944, 0.7597073032343125, 0.7595883097670768, 0.7597878013745345, 0.7596031342935415, 0.7596636347633963, 0.7597012916860032]

# TODO: Compare the predictions with the actual values, also calculates the mean squared error!
"""
