# This script implements the softmax logistic regression for multiclass classification.
# Starting in version 1.7, LogisticRegression will always use the "multinomial" approach
# for multi-class classification. This change is aimed at simplifying the API
# and possibly improving performance and consistency, as "multinomial" tends to be more robust
# for multi-class tasks.
#
# Notice the 3d scatter plot in the end.
# There you can see the decision boundary.
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
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
    x = MinMaxScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LogisticRegression(multi_class="multinomial")
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {acc}")

    # Notice in the theta, we have a pair of values for each class.
    # And one intercept per class as well
    # But all of these are calculated as a single model instead.
    print(f"Theta: {model.coef_} with intercept: {model.intercept_}")

    # Print the 3dplot scatter plot and the cost curve
    show_3d_scatter(
        x_test[:100, 0].flatten(),
        x_test[:100, 1].flatten(),
        y_test[:100].flatten(),
        model,
    )
