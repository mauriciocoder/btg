# https://datahub.io/core/sea-level-rise
import pandas as pd
import matplotlib as plt
import os
from datetime import datetime
import numpy as np
import seaborn as sns
from scipy import stats


def save_fig(fig: plt.pyplot.Figure, filename: str):
    output_dir = "../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"
    fig.savefig(os.path.join(output_dir, filename))


def load_sea_level_data() -> pd.DataFrame:
    df = (
        pd.read_csv(
            "../../data/sea_level_data.csv",
            usecols=[
                "Year",
                "CSIRO Adjusted Sea Level",
                "Lower Error Bound",
                "Upper Error Bound",
            ],
        )
        .sort_values(by="Year")
        .set_index("Year")
    )
    df.dropna(axis=0, subset=["CSIRO Adjusted Sea Level"], inplace=True)
    df["Sea Level Diff"] = df["CSIRO Adjusted Sea Level"].diff()
    return df


def hide_year_ticks(ax: plt.axes.Axes) -> None:
    # Just hiding some x labels
    for tick_label in ax.get_xticklabels():
        position = tick_label.get_position()[0]
        if position % 10 != 0:
            tick_label.set_visible(False)


def plot_bar_chart(df: pd.DataFrame) -> None:
    fig, ax = plt.pyplot.subplots(figsize=(16, 8))
    sns.barplot(data=df, x="Year", y="CSIRO Adjusted Sea Level", ax=ax)
    hide_year_ticks(ax)
    save_fig(fig, "sea_level_bar_chart")


def plot_line_chart(df: pd.DataFrame) -> None:
    fig, ax = plt.pyplot.subplots(figsize=(16, 8))
    sns.lineplot(df, x="Year", y="Sea Level Diff", ax=ax)
    # Set y-axis range
    ax.set_ylim(-1, 1)
    # Draw a horizontal line
    ax.axhline(y=0.0, color="red", linestyle="--", label="Horizontal Line")
    hide_year_ticks(ax)
    save_fig(fig, "diff_line_chart")


def plot_scatter_chart(df: pd.DataFrame) -> None:
    fig, ax = plt.pyplot.subplots(figsize=(16, 8))
    sns.scatterplot(data=df, x="Year", y="CSIRO Adjusted Sea Level", ax=ax)
    hide_year_ticks(ax)
    # Regression line extracted from 1880, 2013
    regress = stats.linregress(
        x=df.index.values, y=df["CSIRO Adjusted Sea Level"].values
    )
    x = np.linspace(1880, 2050)
    line = pd.DataFrame({"x": x, "y": regress.slope * x + regress.intercept})
    sns.lineplot(
        data=line,
        x="x",
        y="y",
        ax=ax,
        color="red",
        label="Tendency considering all data",
    )
    # Regression line extracted from 2000, 2013
    regress2 = stats.linregress(
        x=df.iloc[-13:].index.values, y=df.iloc[-13:]["CSIRO Adjusted Sea Level"].values
    )
    x = np.linspace(2000, 2050)
    line = pd.DataFrame({"x": x, "y": regress2.slope * x + regress2.intercept})
    sns.lineplot(
        data=line,
        x="x",
        y="y",
        ax=ax,
        color="purple",
        label="Tendency considering 2000 onwards",
    )

    # The x label should be Year, the y label should be Sea Level (inches), and the title should be Rise in Sea Level.
    ax.set_title("Rise in Sea Level", fontsize=20, fontweight="bold")
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("Sea Level (inches)", fontsize=16)

    # Optional: Add a legend
    ax.legend()

    save_fig(fig, "scatter_chart")


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    df = load_sea_level_data()
    print(df.info())
    print(df.describe(include="all"))
    plot_bar_chart(df)
    plot_scatter_chart(df)
    plot_line_chart(df)
