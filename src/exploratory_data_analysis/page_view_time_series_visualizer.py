# https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/page-view-time-series-visualizer
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import seaborn as sns


def save_fig(fig: plt.Figure, filename: str):
    output_dir = "../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"
    fig.savefig(os.path.join(output_dir, filename))


def load_visitors_data():
    df = pd.read_csv("../../data/fcc-forum-pageviews.csv", index_col="date")
    df = df[
        (df["value"] > df["value"].quantile(0.025))
        & (df["value"] < df["value"].quantile(0.975))
    ]
    # Split the date into columns
    df_index_split = df.index.to_series().str.split("-", expand=True)
    df_index_split.columns = ["year", "month", "day"]
    df = df.join(df_index_split)
    return df


def draw_bar_plot(df: pd.DataFrame) -> None:
    # Extract a Series of averages indexed by (year, month) as numbers
    averages = df.groupby(["year", "month"])["value"].mean()
    averages.index = averages.index.map(lambda x: (int(x[0]), int(x[1])))

    fig, axes = plt.subplots(figsize=(12, 6))
    bar_width = 0.3
    months_style = {
        1: {"name": "January", "color": "#1f77b4"},
        2: {"name": "February", "color": "#ff7f0e"},
        3: {"name": "March", "color": "#2ca02c"},
        4: {"name": "April", "color": "#d62728"},
        5: {"name": "May", "color": "#9467bd"},
        6: {"name": "June", "color": "#8c564b"},
        7: {"name": "July", "color": "#e377c2"},
        8: {"name": "August", "color": "#7f7f7f"},
        9: {"name": "September", "color": "#bcbd22"},
        10: {"name": "October", "color": "#17becf"},
        11: {"name": "November", "color": "#aec7e8"},
        12: {"name": "December", "color": "#ffbb78"},
    }
    # Positions for the bars
    reference = np.array([0, 5, 10, 15])
    for offset_index, month in enumerate(range(1, 13)):
        # load the averages for each month
        m = averages.xs(month, level="month")
        # Plot a set of bars (each set has bars of the same month but different years)
        axes.bar(
            reference[-len(m.values) :] + (offset_index * bar_width),
            m.values,
            color=months_style[month]["color"],
            width=bar_width,
            edgecolor="black",
            label=months_style[month]["name"],
        )
    # Add labels and title
    axes.set_xlabel("Years")
    axes.set_ylabel("Average Page Views")
    axes.set_xticks(reference + ((bar_width * 12) / 2))
    axes.set_xticklabels(df["year"].unique())
    axes.legend()

    save_fig(fig, "barplot")


def draw_line_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(figsize=(24, 12))
    axes.plot(df.index.values, df["value"], color="red", linewidth=1)
    axes.set_xlabel("Date")
    axes.set_ylabel("Page Views")
    axes.set_title("Daily freeCodeCamp Forum Page Views 5/2016-12/2019")

    # Set x-axis ticks to only show 20 entries
    num_ticks = 20
    xticks = df.index.values[:: len(df) // num_ticks]
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticks, rotation=45, ha="right")

    save_fig(fig, "page_views")


def draw_box_plot(df: pd.DataFrame) -> None:
    df_tmp = pd.DataFrame(df)
    sns.set_theme(style="dark")
    fig, axes = plt.subplots(figsize=(18, 6), ncols=2)
    df_sorted_by_year = df_tmp.sort_values(by="year")
    sns.boxplot(
        x=df_sorted_by_year["year"],
        y=df_sorted_by_year["value"],
        native_scale=True,
        ax=axes[0],
    )
    axes[0].set_title("Year-wise Box Plot (Trend)")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Page Views")
    df_sorted_by_month = df_tmp.sort_values(by="month")
    months = {
        "01": "Jan",
        "02": "Feb",
        "03": "Mar",
        "04": "Apr",
        "05": "May",
        "06": "Jun",
        "07": "Jul",
        "08": "Aug",
        "09": "Sep",
        "10": "Oct",
        "11": "Nov",
        "12": "Dec",
    }
    df_sorted_by_month["month"] = df_sorted_by_month["month"].map(months)
    sns.boxplot(
        x=df_sorted_by_month["month"],
        y=df_sorted_by_month["value"],
        native_scale=True,
        ax=axes[1],
    )
    axes[1].set_title("Month-wise Box Plot (Seasonality)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Page Views")

    save_fig(fig, "boxplot")


if __name__ == "__main__":
    df = load_visitors_data()
    print(df)
    draw_line_plot(df)
    draw_bar_plot(df)
    draw_box_plot(df)
