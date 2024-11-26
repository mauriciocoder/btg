# File for assignment https://mlcourse.ai/book/topic04/assignment04_regression_wine.html
# Notice how LassoCV can train linear models by adding some regularization. We can observe inspecting
# the coefficients (parameters trained) which feature has the biggest impact on the model.
# Maybe this could be used to predict to remove some of these features.
# Notice that data exploration will remove some of these features based on the correlation between different features.

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error
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


def train_linear_regression_model(
    feature_names, x_train_scaled, y_train, x_test_scaled, y_test
):
    print("### Training a linear regression model")
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)
    y_predict_train = model.predict(x_train_scaled)
    mse = mean_squared_error(y_train, y_predict_train)
    print(f"Mean squared error (train): {mse:.3f}")
    y_predict_test = model.predict(x_test_scaled)
    mse = mean_squared_error(y_test, y_predict_test)
    print(f"Mean squared error (test): {mse:.3f}")
    print_coefficients(model, feature_names)


def train_lasso_model(feature_names, x_train_scaled, y_train, x_test_scaled, y_test):
    print("### Training a lasso model")
    model = LassoCV(
        cv=5, random_state=17
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


if __name__ == "__main__":
    df = pd.read_csv("../../../data/winequality-white.csv")
    print(df.head())
    print(df.info())

    y = df["quality"]
    x = df.drop(columns="quality")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.7, random_state=17
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    feature_names = list(x.columns)
    train_linear_regression_model(
        feature_names, x_train_scaled, y_train, x_test_scaled, y_test
    )
    train_lasso_model(feature_names, x_train_scaled, y_train, x_test_scaled, y_test)
