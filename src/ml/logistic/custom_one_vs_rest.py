# This script implements the one-vs-all logistic regression for multiclass classification.
# Manually trained a single scikit learn model for each classification value in this dataset (0, 1, 2).
# This is not needed but implemented for illustration purposes. Notice the 3d scatter plot in the end.
# There you can see the decision boundary
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from utils.plot import save_fig, show_3d_scatter

# Required for live plot
# requires:sudo apt-get install python3-pyqt5
# pip install PyQt5
matplotlib.use("Qt5Agg")  # or 'Qt5Agg', 'MacOSX', etc.
# from utils.plot import save_fig, plot_linechart

if __name__ == "__main__":
    df = pd.read_csv(
        "../../../data/classification-multiclass.csv",
        dtype={"x": "float32", "y": "float32", "z": "int8"},
    )
    x = df[["x", "y"]].values
    y = df["z"].values.reshape(df["z"].shape[0], 1)
    # One hot encoding
    # y_one_hot now holds the probability of each of the classes
    y_one_hot = OneHotEncoder().fit_transform(y).toarray()
    x = MinMaxScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.2)
    models = []
    for i in range(y_one_hot.shape[1]):
        model = LogisticRegression()
        model.fit(x_train, y_train[:, i])
        model.class_label = i
        models.append(model)
        print(f"Model-{i} - coeff: {model.coef_} - intercept: {model.intercept_}")
    results = []
    for i, model in enumerate(models):
        y_predict = model.predict(x_test)
        results.append(accuracy_score(y_test[:, i], y_predict))
        show_3d_scatter(
            x_test[:, 0].flatten(),
            x_test[:, 1].flatten(),
            y_test[:, i].flatten(),
            model,
        )
    print(results)
