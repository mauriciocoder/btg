# https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction


import sys
from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


def save_fig(fig: plt.Figure, filename: str):
    output_dir = "../../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"
    full = os.path.join(output_dir, filename)
    fig.savefig(full)


def calculate_figsize(tree_depth: int, max_nodes: int) -> tuple[int, int]:
    # Calculate the width and height of the figure
    width = max_nodes * (tree_depth + 1)
    height = tree_depth * 2
    # Scale down the size to fit within a reasonable range
    scale = 0.5
    width *= scale
    height *= scale

    return (width, height)


def plot_decision_tree(
    tree: DecisionTreeClassifier, feature_names: list[str], filename: str
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(
        figsize=calculate_figsize(tree.get_depth(), tree.get_n_leaves())
    )
    plot_tree(
        tree,
        filled=True,
        feature_names=feature_names,
        rounded=True,
        ax=ax,
    )
    save_fig(fig, filename)
    return fig, ax


def load_taiwanese_bankruptcy_data() -> pd.DataFrame:
    df = pd.read_csv("../../../data/taiwanese-bankruptcy-prediction.csv")
    scaler = MinMaxScaler()
    y = df["Bankrupt"]
    df.pop("Bankrupt")

    df_normalized = scaler.fit_transform(df)
    df = pd.DataFrame(df_normalized, columns=df.columns)
    df["Bankrupt"] = y

    return df


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    df = load_taiwanese_bankruptcy_data()
    print("Distribution:")
    print(df["Bankrupt"].value_counts(normalize=True))

    # Split the data into 80% training and 20% test
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop("Bankrupt", axis=1), df["Bankrupt"], test_size=0.2, random_state=42
    )

    tree = DecisionTreeClassifier(
        criterion="entropy", max_depth=5, min_samples_leaf=100, random_state=42
    )
    tree.fit(x_train, y_train)
    param_grid = {
        "min_samples_leaf": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        "criterion": ["entropy", "gini"],
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 30, 40, 50],
    }
    grid_search = GridSearchCV(tree, param_grid, cv=5, n_jobs=-1, verbose=3)

    # Fit the GridSearchCV object to the data
    grid_search.fit(x_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    y_predicted = grid_search.predict(x_test)
    accuracy_score = accuracy_score(y_test, y_predicted)
    print(f"Accuracy score: {accuracy_score}")

    best_tree = grid_search.best_estimator_
    plot_decision_tree(best_tree, x_train.columns, "decision_tree_plot_bankruptcy")
