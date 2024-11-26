# This script implements the one-vs-all logistic regression for multiclass classification.
# The scikit learning implementation OneVsRestClassifier encapsulates the one-hot-encoding.
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
        "../../../data/winequality-white.csv",
        dtype={"x": "float32", "y": "float32", "z": "int8"},
    )
    x = df.drop(columns="quality")
    y = df["quality"].values.reshape(x.shape[0], 1)
    x = MinMaxScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = OneVsRestClassifier(LogisticRegression())
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print(f"Accuracy: {acc}")

    # Calculate residuals
    residuals = y_test[0] - y_predict
    print(f"Residuals = 0: {np.sum(residuals == 0)}")
    print(f"Residuals = 1: {np.sum(abs(residuals) == 1)}")
    print(f"Residuals = 2: {np.sum(abs(residuals) == 2)}")
    print(f"Residuals = 3: {np.sum(abs(residuals) == 3)}")

    # Calculate standard deviation of residuals
    std_dev = np.std(residuals)
    print("Standard Deviation of Residuals:", std_dev)
    print("Test size:", len(y_test))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    ax.hist(np.abs(residuals), bins=100)
    plt.show()
