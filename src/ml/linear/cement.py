# Experiments to predict concrete compressive strength
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def print_coefficients(model, feature_names):
    feature_coefficients = dict(zip(feature_names, model.coef_))
    feature_coefficients = dict(
        sorted(feature_coefficients.items(), key=lambda item: item[1])
    )
    # Display the results
    print("Feature coefficients:")
    for feature, coef in feature_coefficients.items():
        print(f"{feature}: {coef}")
    print("Intercept:", model.intercept_)  # intercept term


def train_lasso_model(
    feature_names, x_train_scaled, y_train, x_test_scaled, y_test
) -> LassoCV:
    print("### Training a lasso model")
    model = LassoCV(
        cv=5
    )  # cv=5 means 5-fold cross-validation. alphas is the regularization strength (I have manually provided some, although LassoCV can do it automatically)
    model.fit(x_train_scaled, y_train)
    y_predict_train = model.predict(x_train_scaled)
    mse = mean_squared_error(y_train, y_predict_train)
    print(f"Alpha: {model.alpha_}")
    print(f"Mean squared error (train): {mse:.3f}")
    y_predict_test = model.predict(x_test_scaled)
    mse = mean_squared_error(y_test, y_predict_test)
    print(f"Mean squared error (test): {mse:.3f}")
    print_coefficients(model, feature_names)
    return model


if __name__ == "__main__":
    df = pd.read_csv("../../../data/concrete.csv")
    print(df.head())
    print(df.info())

    # TODO: It is worth creating a concrete calculator having the sum of all components as a limit
    # df["Weight"] = df.drop(columns=["Strength"]).sum(axis=1)
    # print(f"Min weight={df["Weight"].min()}")
    # print(f"Max weight={df["Weight"].max()}")

    # This is giving the idea oh how each component affects the strength
    """
    df["Robust"] = df["Strength"].apply(
        lambda x: 1 if x > 80.00 else (-1 if x <= 20.00 else 0)
    )
    """
    # grouped = df.groupby(by="Robust").mean()
    # grouped.to_csv("../../../results/grouped_concrete_data.csv")

    x = df.drop(columns=["Strength"]).values
    y = df["Strength"].values
    x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    """
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)
    """
    """
    # No scaler
    x_train_scaled = x_train
    x_test_scaled = x_test
    """

    column_names = df.drop(columns=["Strength"]).columns
    model = train_lasso_model(
        column_names, x_train_scaled, y_train, x_test_scaled, y_test
    )

    predicted = model.predict(x_test_scaled)
    print("Predicted:", predicted)
    print("Actual:", y_test)

    r2 = r2_score(y_test, predicted)
    print("R^2 score:", r2)
