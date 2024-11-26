# In this file I train some models based on my custom implementation of linear regression (both gradient descent and normal equation)
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from custom_regressor import LinearRegressor, Method

if __name__ == "__main__":
    df = pd.read_csv("../../../data/winequality-white.csv")
    print(df.head())
    print(df.info())

    y = df["quality"]
    # Initialize the scaler to scale to the [0, 1] range
    scaler = MinMaxScaler()
    # Apply the scaler to the DataFrame (excluding target columns if necessary)
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    x = df.drop(columns="quality")
    # Split the data into 80% training and 20% test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    # Custom model regression using normal equation
    print("Custom model regression using normal equation")
    model = LinearRegressor(method=Method.NORMAL_EQUATION)
    model.fit(x_train.values, y_train.values)
    y_predict = model.predict(x_test.values)
    result = DataFrame({"y_test": y_test, "y_predict": y_predict})
    error = mean_squared_error(y_test, y_predict)
    evs = explained_variance_score(y_test, y_predict)
    print(f"mean_squared_error={error}, evs={evs}")

    print("Custom model regression using batch gradient descent")
    model = LinearRegressor(method=Method.BATCH)
    model.fit(x_train.values, y_train.values)
    y_predict = model.predict(x_test.values)
    result = DataFrame({"y_test": y_test, "y_predict": y_predict})
    error = mean_squared_error(y_test, y_predict)
    evs = explained_variance_score(y_test, y_predict)
    print(f"mean_squared_error={error}, evs={evs}")

    print("SciKit learn model")
    model = LinearRegression()
    model.fit(x_train.values, y_train.values)
    y_predict = model.predict(x_test.values).round()
    error = mean_squared_error(y_test, y_predict)
    evs = explained_variance_score(y_test, y_predict)
    print(f"mean_squared_error={error}, evs={evs}")
    print(f"Accuracy: {accuracy_score(y_test, y_predict)}")
