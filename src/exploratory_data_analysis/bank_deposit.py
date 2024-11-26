# https://archive.ics.uci.edu/dataset/222/bank+marketing
import os
from datetime import datetime
import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np


def load_bank_data() -> pd.DataFrame:
    type = {
        "age": np.int8,
        "balance": np.int32,
        "day": np.int8,
        "duration": np.int16,
        "campaign": np.int8,
        "pdays": np.int16,
        "previous": np.int16,
    }
    df = pd.read_csv("../../data/bank-full.csv", delimiter=";", dtype=type)
    for col in ["default", "housing", "loan", "y"]:
        df[col] = df[col].map({"yes": True, "no": False}).astype(np.bool)
    df["contact"] = (
        df["contact"].map({"unknown": 0, "telephone": 1, "cellular": 2}).astype(np.int8)
    )
    months_to_integers = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    df["month"] = df["month"].map(months_to_integers).astype(np.int8)
    poutcome = {"unknown": 0, "failure": 1, "nonexistent": 2, "other": 3, "success": 4}
    df["poutcome"] = df["poutcome"].map(poutcome).astype(np.int8)
    education = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
    df["education"] = df["education"].map(education).astype(np.int8)
    marital = {"single": 0, "divorced": 1, "married": 2}
    df["marital"] = df["marital"].map(marital).astype(np.int8)
    return df


def save_fig(fig: plt.pyplot.Figure, filename: str):
    output_dir = "../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"
    fig.savefig(os.path.join(output_dir, filename))


def plot_histograms(df: pd.DataFrame, columns: list[str], hue: str = None) -> None:
    fig, axes = plt.pyplot.subplots(nrows=len(columns), ncols=1, figsize=(8, 42))
    for i, col in enumerate(columns):
        if hue:
            sns.histplot(df, x=col, hue=hue, multiple="dodge", ax=axes[i])
        else:
            sns.histplot(df, x=col, ax=axes[i])
        axes[i].set_title(f"Histogram of {col}")
    fig.tight_layout()
    save_fig(fig, "histograms")


def plot_heatmap(df: pd.DataFrame, method: str = "pearson") -> None:
    # Create a figure and axis
    fig, ax = plt.pyplot.subplots(figsize=(16, 8))
    corr_matrix = df.corr(method=method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    g = sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax, mask=mask
    )
    g.set_title(f"Correlation Matrix Heatmap - {method} Method")
    save_fig(fig, f"correlation_{method}")
    plt.pyplot.clf()


def get_boxplot_range(series: pd.Series) -> tuple[int, int]:
    # Calculate the IQR and determine limits
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    # Define limits (excluding outliers)
    lower_limit = max(Q1 - 1.5 * IQR, series.min())
    upper_limit = min(Q3 + 1.5 * IQR, series.max())
    return lower_limit, upper_limit


def plot_box_plot(df: pd.DataFrame, columns: list[str], hue: str) -> None:
    sns.set_theme(style="dark")
    fig, axes = plt.pyplot.subplots(figsize=(18, 36), nrows=len(columns))
    axes_count = 0
    for column in columns:
        sns.boxplot(data=df, x=column, hue=hue, ax=axes[axes_count])
        axes[axes_count].set_title(f"{column} Box Plot")
        axes[axes_count].set_xlabel(column)
        r1 = get_boxplot_range(df[df[hue] == True][column])
        r2 = get_boxplot_range(df[df[hue] == False][column])
        # Set x-axis limits based on the calculated range
        axes[axes_count].set_xlim(
            r1[0] if r1[0] < r2[0] else r2[0], r1[1] if r1[1] > r2[1] else r2[1]
        )
        axes_count += 1
    fig.tight_layout()
    save_fig(fig, "boxplot")


if __name__ == "__main__":
    df = load_bank_data()
    categorical = [
        "age",
        "day",
        "previous",
        "contact",
        "month",
        "poutcome",
        "education",
        "marital",
        "default",
        "housing",
        "loan",
    ]
    # plot_histograms(df, columns=categorical, hue="y")
    categorical.append("y")
    # plot_heatmap(df[categorical], method="pearson")
    # plot_heatmap(df[categorical], method="spearman")
    numerical = ["pdays", "campaign", "balance", "duration", "previous"]
    # plot_histograms(df, numerical, hue="y")
    # plot_box_plot(df, columns=numerical, hue="y")
    numerical.append("y")
    # plot_heatmap(df[numerical], method="pearson")
    # plot_heatmap(df[numerical], method="spearman")

    print("###### Starting the analysis:")
    pd.set_option("display.max_columns", None)
    print(df["y"].value_counts(normalize=True))
    # print(df["y"].info())
    print("###### Cross Tab")
    df["duration_target"] = (df["duration"] <= 725) & (df["duration"] >= 244)
    print(pd.crosstab(df["y"], df["duration_target"], margins=True, normalize=True))
    df["poutcome_4"] = df["poutcome"] == 4
    print(pd.crosstab(df["y"], df["poutcome_4"], margins=True, normalize=True))

    df["poutcome_duration"] = df["poutcome_4"] & df["duration_target"]
    print(df[["y", "poutcome_4", "duration_target", "poutcome_duration"]])
    print(pd.crosstab(df["y"], df["poutcome_duration"], margins=True, normalize=True))

    print("###### Group By")
    print(df.groupby(by=["y"])["duration"].describe())
    print(df.groupby(by=["y"])["poutcome"].describe())
    print(df.groupby(by=["y"])["balance"].describe())
