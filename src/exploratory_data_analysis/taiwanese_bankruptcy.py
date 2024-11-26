# https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction


import sys
from typing import Literal

import pandas as pd
import matplotlib as plt
import os
from datetime import datetime
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


def save_fig(fig: plt.pyplot.Figure, filename: str):
    output_dir = "../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"
    full = os.path.join(output_dir, filename)
    fig.savefig(full)


def load_taiwanese_bankruptcy_data() -> pd.DataFrame:
    df = pd.read_csv(
        "../../data/taiwanese-bankruptcy-prediction.csv",
    )
    scaler = MinMaxScaler()
    y = df["Bankrupt"]
    df.pop("Bankrupt")
    print("Scaling the data...")
    df_normalized = scaler.fit_transform(df)
    df = pd.DataFrame(df_normalized, columns=df.columns)
    df["Bankrupt"] = y

    lower_quantile = df.quantile(0.025)
    df["cols_below_lower_quantile"] = (df < lower_quantile).sum(axis=1)

    higher_quantile = df.quantile(0.975)
    df["cols_above_higher_quantile"] = (df > higher_quantile).sum(axis=1)

    return df


def plot_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = None,
    figsize: tuple = (12, 8),
    filename: str = f"boxplot",
    orient: str = "h",
) -> None:
    fig, axes = plt.pyplot.subplots(figsize=figsize)
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        ax=axes,
        orient=orient,
    )
    save_fig(fig, filename)


def plot_histograms(
    df: pd.DataFrame, columns: list[str], hue: str = None, filename: str = "histograms"
) -> None:
    print(f"Ploting histograms for {len(columns)} columns")
    fig, axes = plt.pyplot.subplots(
        nrows=len(columns), ncols=1, figsize=(8, 8 * len(columns))
    )
    for i, col in enumerate(columns):
        ax = axes[i] if len(columns) > 1 else axes
        print(f"Generating plot for {col}")
        if hue:
            sns.histplot(
                df,
                x=col,
                hue=hue,
                multiple="dodge",
                ax=ax,
                bins="fd",
                kde=True,
            )
        else:
            sns.histplot(df, x=col, ax=ax)
        ax.set_title(f"Histogram of {col}")
    fig.tight_layout()
    save_fig(fig, filename)


def plot_corr_matrix(
    df: pd.DataFrame,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    filename: str = "corr_matrix",
) -> None:
    num_cols = len(df.columns)
    fig, axes = plt.pyplot.subplots(figsize=(num_cols * 2, num_cols * 2))
    corr = df.corr(method=method)
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=axes)
    plt.pyplot.title("Correlation Matrix Heatmap")
    save_fig(fig, filename)


def filter_df_based_on_correlations(df: pd.DataFrame) -> pd.DataFrame:
    # List of columns we found that has correlation with Bankrupt
    # First correlation filter
    selected_columns = [
        "Bankrupt",
        " ROA(C) before interest and depreciation before interest",
        " ROA(A) before interest and % after tax",
        " ROA(B) before interest and depreciation after tax",
        " Net Value Per Share (A)",
        " Net Value Per Share (B)",
        " Net Value Per Share (C)",
        " Persistent EPS in the Last Four Seasons",
        " Operating Profit Per Share (Yuan ¥)",
        " Per Share Net profit before tax (Yuan ¥)",
        " Debt ratio %",
        " Net worth per Assets",
        " Borrowing dependency",
        " Operating profit per Paid-in capital",
        " Net profit before tax per Paid-in capital",
        " Working Capital to Total Assets",
        " Current Liability to Assets",
        " Working Capital per Equity",
        " Current Liabilities per Equity",
        " Retained Earnings to Total Assets",
        " Total expense per Assets",
        " Current Liability to Equity",
        " Equity to Long-term Liability",
        " CFO to Assets",
        " Current Liability to Current Assets",
        " Liability-Assets Flag",
        " Net Income to Total Assets",
        " Net Income to Stockholder's Equity",
        " Liability to Equity",
        "cols_below_lower_quantile",
    ]

    # Second correlation filter
    selected_columns = [
        "Bankrupt",
        " ROA(A) before interest and % after tax",
        " Net Value Per Share (A)",
        " Debt ratio %",
        " Operating profit per Paid-in capital",
        " Working Capital to Total Assets",
        " Current Liability to Assets",
        " Working Capital per Equity",
        " Retained Earnings to Total Assets",
        " Total expense per Assets",
        " Equity to Long-term Liability",
        " CFO to Assets",
        " Current Liability to Current Assets",
        " Liability-Assets Flag",
        " Net Income to Stockholder's Equity",
        " Liability to Equity",
        "cols_below_lower_quantile",
    ]

    return df.loc[:, selected_columns]


def print_grouped(by: str, columns: list[str], df: pd.DataFrame) -> None:
    grouped = df.groupby([by])
    # Define aggregation functions
    agg_funcs = {
        col: [
            "mean",
            "std",
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.5),
            lambda x: x.quantile(0.75),
        ]
        for col in columns
    }

    # Perform aggregation
    result = grouped.agg(agg_funcs)
    # Rename the columns for clarity
    # Create a multi-level column index for the result DataFrame
    result.columns = pd.MultiIndex.from_product(
        [
            columns,
            ["Mean", "Std", "25th Percentile", "50th Percentile", "75th Percentile"],
        ],
        names=["Metric", "Stat"],
    )
    result.to_csv("../../results/grouped_bankrupt_data_with_quantiles.csv", index=False)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    df = load_taiwanese_bankruptcy_data()
    print("Distribution:")
    print(df["Bankrupt"].value_counts(normalize=True))
    # print(df.info())
    # print(df.describe(include="all"))
    """
    plot_corr_matrix(
        df[["cols_below_lower_quantile", "cols_above_higher_quantile", "Bankrupt"]],
        method="pearson",
        filename="bankrupt/corr_matrix_bankrupt_quantiles",
    )
    plot_corr_matrix(
        df,
        method="pearson",
        filename="bankrupt/corr_matrix_bankrupt_all",
    )
    """
    """
    group_by_df = df.groupby("Bankrupt")[
        ["cols_below_lower_quantile", "cols_above_higher_quantile"]
    ].agg(["mean", "std", "min", "max", "sum", "count", "median"])
    print(group_by_df)
    group_by_df.to_csv(
        "../../results/grouped_bankrupt_data_with_quantiles.csv", index=False
    )
    """
    """
    for col in ["cols_below_lower_quantile", "cols_above_higher_quantile"]:
        plot_boxplot(df, x=col, y="Bankrupt", filename=f"bankrupt/{col}_boxplot")
    """
    df = filter_df_based_on_correlations(df)
    """
    # Generating boxplot for filtered columns
    for col in df.columns:
        plot_boxplot(
            df, x=col, y="Bankrupt", filename=f"bankrupt/second_filtered_{col}_boxplot"
        )
    # Generating heatmap for filtered columns
    plot_corr_matrix(
        df,
        method="pearson",
        filename="bankrupt/second_filtered_corr_matrix_bankrupt",
    )
    """
    # print_grouped("Bankrupt", df.columns, df)

    # first mask using
    # Debt ration %
    # ROA(A)
    # cols_below_lower_quantile
    mask1 = df[" Debt ratio %"] > 0.187426308310911
    mask2 = df["cols_below_lower_quantile"] >= 6
    mask3 = df[" ROA(A) before interest and % after tax"] < 0.537723506323594

    df["high_debt_ratio"] = mask1
    print(
        pd.crosstab(df["Bankrupt"], df["high_debt_ratio"], margins=True, normalize=True)
    )
    df["high_cols_below_lower_quantile"] = mask2
    print(
        pd.crosstab(
            df["Bankrupt"],
            df["high_cols_below_lower_quantile"],
            margins=True,
            normalize=True,
        )
    )
    df["low_roa"] = mask3
    print(
        pd.crosstab(
            df["Bankrupt"],
            df["low_roa"],
            margins=True,
            normalize=True,
        )
    )
    df["all_mask"] = mask1 & mask2 & mask3
    print(
        pd.crosstab(
            df["Bankrupt"],
            df["all_mask"],
            margins=True,
            normalize=True,
        )
    )

"""
Correlations:
bankrupt x cols_below_lower_quantile = 0.32
bankrupt x Net Income to total assets = -0.32
bankrupt x roa(c) = -0.28
bankrupt x roa(b) = -0.27
bankrupt x roa(a) = -0.26
bankrupt x Debt ratio % = 0.25
bankrupt x Net worth per Assets = -0.25
bankrupt x persistent EPS in the last four Seasons = -0.22
bankrupt x Retained Earnings to Total assets = -0.22
bankrupt x Net profit before tax per Paid-in capital = -0.21
bankrupt x Per share Net profit before tax = -0.2
bankrupt x Working capital to total assets = -0.19
bankrupt x Current liability to Assets = 0.19
bankrupt x Borrowing dependency = 0.18
bankrupt x Net Income to stockholders equity = -0.18
bankrupt x liability to equity = -0.17
bankrupt x Net value per share (a) = -0.17
bankrupt x Net value per share (b) = -0.16
bankrupt x Net value per share (c) = -0.16
bankrupt x Working capital per equity = -0.15
bankrupt x Current liability per equity = 0.15
bankrupt x Current liability to Equity = 0.15
bankrupt x Total expenses per asset = 0.14
bankrupt x Operating profit per share = -0.14
bankrupt x Operating profit per paid-in capital = -0.14
bankrupt x liability-assets flag = 0.14
bankrupt x Equity to Long-term liability = 0.14
bankrupt x CFO to assets = -0.12
"""

"""
# TODO: Generate more graphs after removing the following columns:
# Selections to remove after next round of filtering and correlation analysis:
"ROA(C)"
"ROA(B)"
"Net Value Per Share (C)"
"Net Value Per Share (B)"
"Per Share Net profit before tax (Yuan)"
"Persistent EPS in the Last Four Seasons"
"Net Worth Per Assets"
"Operating profit per share (Yuan)"
"Net profit before tax per paid-in capital (Yuan)"
"Borrowing dependency"
"Current liability per Equity"
"Current liability to Equity"
"""
