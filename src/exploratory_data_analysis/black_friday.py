# https://www.kaggle.com/datasets/pranavuikey/black-friday-sales-eda/data
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
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_black_friday_data() -> pd.DataFrame:
    type = {
        "User_ID": np.int32,
        "Occupation": np.int8,
        "Marital_Status": np.int8,
        "Product_Category_1": np.int8,
        "Purchase": np.int16,
        "Product_ID": np.dtype("S9"),
    }
    df = pd.read_csv("../../data/black_friday.csv", dtype=type)
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
    df["Product_Category"] = (
        df["Product_Category_1"].astype(str)
        + "|"
        + df["Product_Category_2"].astype(str)
    )
    df["Product_Category_Extended"] = (
        df["Product_Category"].astype(str) + "|" + df["Product_Category_3"].astype(str)
    )
    df["Demographic"] = (
        df["Gender"].astype(str)
        + "|"
        + df["Age"].astype(str)
        + "|"
        + df["Marital_Status"].astype(str)
    )
    df["Demographic_Extended"] = (
        df["Demographic"].astype(str)
        + "|"
        + df["Occupation"].astype(str)
        + "|"
        + df["City_Category"].astype(str)
        + "|"
        + df["Stay_In_Current_City_Years"].astype(str)
    )
    df.drop(["User_ID", "Product_ID"], axis=1, inplace=True)
    return df


def save_fig(fig: plt.Figure, filename: str):
    output_dir = "../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"
    fig.savefig(os.path.join(output_dir, filename))


def plot_histograms(df: pd.DataFrame, columns: list[str], hue: str = None) -> None:
    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(8, 42))
    for i, col in enumerate(columns):
        if hue:
            sns.histplot(df, x=col, hue=hue, multiple="dodge", ax=axes[i])
        else:
            sns.histplot(df, x=col, ax=axes[i])
        axes[i].set_title(f"Histogram of {col}")
    fig.tight_layout()
    save_fig(fig, "histograms")


def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str = None) -> None:
    fig, axes = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, style=hue, ax=axes)
    save_fig(plt.gcf(), f"scatterplot_{y}_vs_{x}")


def plot_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = None,
    label: str = None,
    figsize: tuple = (12, 8),
) -> None:
    fig, axes = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        ax=axes,
        showfliers=False,
        order=np.sort(df[x].unique()),
    )
    save_fig(fig, f"{label}_boxplots")


def plot_heatmap(df: pd.DataFrame, method: str = "pearson") -> None:
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(16, 8))
    corr_matrix = df.corr(method=method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    g = sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax, mask=mask
    )
    g.set_title(f"Correlation Matrix Heatmap - {method} Method")
    save_fig(fig, f"correlation_{method}")
    plt.clf()


def plot_pie_chart(df: pd.DataFrame, column: str, color_map: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    counts = df[column].value_counts().sort_index(ascending=True)
    # Apply the colors in the correct order based on the category
    colors = [color_map[category] for category in counts.index]
    counts.plot(kind="pie", autopct="%.1f%%", colors=colors, ax=ax)
    ax.set_title(label)
    ax.legend(title="Categories", labels=counts.index, loc="best")
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_ylabel("")  # Hide the y-label
    save_fig(fig, f"{label}_pie_chart")


if __name__ == "__main__":
    df = load_black_friday_data()
    print(df.shape)
    print(df.info())
    """
    plot_histograms(
        df,
        [
            "Age",
            "City_Category",
            "Gender",
            "Stay_In_Current_City_Years",
            "Marital_Status",
            "Occupation",
            "Purchase",
        ],
    )
    """
    # plot_scatter(df, "Age", "Purchase", hue="Gender")
    # plot_heatmap(df)
    print("###### Group By")
    print(df.groupby(by=["Age"])["Purchase"].describe())

    pd.set_option("display.max_columns", None)
    r = df.groupby(by=["Product_Category_1"])["Purchase"].describe()
    filtered_result = r[["mean", "std", "min", "max"]]
    print(filtered_result)

    print(df.groupby(by=["Age"])["Product_Category_1"].describe())
    print(df.groupby(by=["Gender"])["Product_Category_1"].describe())
    print(df.groupby(by=["Occupation"])["Product_Category_1"].describe())

    plot_boxplot(
        df, x="Product_Category_1", y="Purchase", label="Purchases_x_Product_Category_1"
    )
    plot_boxplot(
        df, x="Gender", y="Product_Category_1", label="Product_Category_1_x_Gender"
    )
    plot_boxplot(df, x="Age", y="Product_Category_1", label="Product_Category_1_x_Age")
    plot_boxplot(
        df,
        x="Stay_In_Current_City_Years",
        y="Product_Category_1",
        label="Product_Category_1_x_Stay_In_Current_City_Years",
    )
    plot_boxplot(
        df,
        x="Marital_Status",
        y="Product_Category_1",
        label="Product_Category_1_x_Marital_Status",
    )
    plot_boxplot(
        df,
        x="City_Category",
        y="Product_Category_1",
        label="Product_Category_1_x_City_Category",
    )
    plot_boxplot(
        df,
        x="Occupation",
        y="Product_Category_1",
        label="Product_Category_1_x_Occupation",
    )
    plot_boxplot(
        df,
        x="Product_Category",
        y="Purchase",
        label="Purchase_x_Product_Category",
        figsize=(64, 8),
    )

    for product_category_1 in df["Product_Category_1"].unique():
        df_product_category_1 = df[df["Product_Category_1"] == product_category_1]
        plot_boxplot(
            df_product_category_1,
            x="Product_Category_1",
            y="Purchase",
            label="Purchase_x_Product_Category_by_demographic",
            hue="Demographic",
            figsize=(32, 8),
        )

    for product_category in df["Product_Category_Extended"].unique():
        print(
            f"###### Purchases per demographic_extended for product_category = {product_category}"
        )
        df_product_category = df[df["Product_Category_Extended"] == product_category]
        r = df_product_category.groupby(by=["Demographic_Extended"])[
            "Purchase"
        ].describe()
        filtered_result = r[["mean", "std", "min", "max"]]
        print(filtered_result)

    # Define categories
    categories = [i for i in range(1, 21)]
    # Generate a color palette with 20 distinct colors
    colors = sns.color_palette("tab20", n_colors=20)
    # Create the color map
    color_map = {category: colors[i] for i, category in enumerate(categories)}
    plot_pie_chart(
        df[df["Gender"] == 1],
        "Product_Category_1",
        color_map=color_map,
        label="product_category_1_male",
    )
    plot_pie_chart(
        df[df["Gender"] == 0],
        "Product_Category_1",
        color_map=color_map,
        label="product_category_1_female",
    )

    # Conclusions:
    # Product_category is almost equally distributed between ages!
    # Product_category is almost equally distributed between gender!
    # Product_category is almost equally distributed between marital_status!
    # Stay_In_Current_City_Years is almost equally distributed between product_category_1
    # There is a pattern between Product_Category_1 and Purchase - A decision tree can be created to predict the purchase
