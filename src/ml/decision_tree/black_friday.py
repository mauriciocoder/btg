# https://www.kaggle.com/datasets/pranavuikey/black-friday-sales-eda/data
# This script finds the best decision tree for predicting the purchase value for a product. And plot a boxplot
# showing the distribution of purchase value differences between trained data and test data.
# Since this is a regression problem, It uses Decision Tree Regressor from scikit-learn.
# The criteria used is the mean squared error.
"""
* Data
* Variable    Definition
* User_ID    User ID - OK!
* Product_ID    Product ID - OK!
* Gender    Sex of User int8 - OK!
        F   0
        M   1
* Age    Age in bins - int8 - OK!
        0-17      0
        18-25     1
        26-35     2
        36-45     3
        46-50     4
        51-55     5
        55+       6
* Occupation    Occupation (Masked) - int8 - OK!
* City_Category    Category of the City (A,B,C) - int8 - OK!
        C   0
        B   1
        A   2
* Stay_In_Current_City_Years    Number of years stay in current city - int8 - OK!
        0   0
        1   1
        2   2
        3   3
        4+  4
* Marital_Status    Marital Status - int8 - OK!
* Product_Category_1    Product Category (Masked) - int8 - OK!
* Product_Category_2    Product may belongs to other category also (Masked) - int8 - OK! - for NAN -1 was set!
* Product_Category_3    Product may belongs to other category also (Masked) - int8 - OK! - for NAN -1 was set!
* Purchase    Purchase Amount (Target Variable) - int16 - OK!
"""


import os
from collections import namedtuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree, DecisionTreeRegressor
import seaborn as sns


TrainedModel = namedtuple(
    "TrainedModel", ["model", "max_depth", "min_samples_leaf", "mse"]
)


def load_black_friday_data() -> pd.DataFrame:
    type = {
        "User_ID": np.int32,
        "Occupation": np.int8,
        "Marital_Status": np.int8,
        "Product_Category_1": np.int8,
        "Purchase": np.int16,
        "Product_ID": np.dtype("S9"),
    }
    df = pd.read_csv("../../../data/black_friday.csv", dtype=type)
    df.drop(columns=["User_ID", "Product_ID"], inplace=True)
    df["Age"] = (
        df["Age"]
        .map(
            {
                "0-17": 0,
                "18-25": 1,
                "26-35": 2,
                "36-45": 3,
                "46-50": 4,
                "51-55": 5,
                "55+": 6,
            }
        )
        .astype(np.int8)
    )
    df["City_Category"] = (
        df["City_Category"].map({"C": 0, "B": 1, "A": 2}).astype(np.int8)
    )
    df["Gender"] = df["Gender"].map({"F": 0, "M": 1}).astype(np.int8)
    df["Stay_In_Current_City_Years"] = (
        df["Stay_In_Current_City_Years"]
        .map({"0": 0, "1": 1, "2": 2, "3": 3, "4+": 4})
        .astype(np.int8)
    )
    df["Product_Category_2"] = df["Product_Category_2"].fillna(-1).astype(np.int8)
    df["Product_Category_3"] = df["Product_Category_3"].fillna(-1).astype(np.int8)
    return df


def save_fig(fig: plt.Figure, filename: str):
    output_dir = "../../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"
    fig.savefig(os.path.join(output_dir, filename), dpi=300)


def plot_decision_tree(
    tree: DecisionTreeRegressor, feature_names: list[str], filename: str
):
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_tree(
        tree,
        filled=True,
        feature_names=feature_names,
        rounded=True,
        ax=ax,
    )
    save_fig(fig, filename, dpi=300)


def plot_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str = None,
    hue: str = None,
    plot_title: str = None,
    filename: str = None,
    figsize: tuple = (12, 8),
) -> None:
    fig, axes = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=axes, showfliers=False)
    fig.suptitle(plot_title)
    save_fig(fig, filename)


if __name__ == "__main__":
    df = load_black_friday_data()
    print(df.shape)
    print(df.info())

    trained_models = []
    x = df.drop(columns=["Purchase"])
    y = df["Purchase"]
    for max_depth in range(15, 40):
        for min_samples_leaf in [10, 20, 50, 100, 200]:
            model = DecisionTreeRegressor(
                criterion="squared_error",
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )
            # Split the data into 80% training and 20% test
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            trained_models.append(TrainedModel(model, max_depth, min_samples_leaf, mse))
    print("Trained Models:")
    trained_models = sorted(trained_models, key=lambda x: x.mse)
    print(trained_models)
    best_model = trained_models[0].model
    print(f"Best Model: {best_model}")

    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    y_pred = best_model.predict(x_test)
    df_compare = pd.DataFrame({"Test Labels": y_test, "Predicted Labels": y_pred})
    df_compare["Diff"] = df_compare["Test Labels"] - df_compare["Predicted Labels"]
    df_compare["Diff_Percent"] = (df_compare["Diff"] / df_compare["Test Labels"]).abs()
    plot_boxplot(
        df_compare,
        x="Diff_Percent",
        plot_title=f"Purchase Regressor - Max Depth: {max_depth} - Min Samples Leaf: {min_samples_leaf} -> MSE: {mse}",
        filename=f"percentual_diff_boxplot_max_depth_{max_depth}_min_samples_leaf_{min_samples_leaf}",
    )
