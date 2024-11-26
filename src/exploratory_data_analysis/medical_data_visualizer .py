# https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/demographic-data-analyzer
from typing import Union

import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np


def load_medical_data() -> pd.DataFrame:
    dtype_mapping = {
        "age": "int32",
        "sex": "int8",
        "cholesterol": "int8",
        "gluc": "int8",
        "height": "int16",
        "weight": "float32",
        "ap_hi": "int16",
        "ap_lo": "int16",
        "smoke": "bool",
        "alco": "bool",
        "active": "bool",
        "cardio": "bool",
    }
    df = pd.read_csv("../../data/medical_examination.csv", dtype=dtype_mapping)
    df.set_index("id", drop=True, inplace=True)
    return df


def clean_medical_data(df: pd.DataFrame) -> pd.DataFrame:
    # Create the overweight column in the df variable
    bmi = df["weight"] / ((df["height"] / 100) ** 2)
    df["overweight"] = bmi > 25
    df["high_cholesterol"] = df["cholesterol"] > 1
    df["high_gluc"] = df["gluc"] > 1
    # df.drop(["cholesterol", "gluc"], axis=1, inplace=True)
    print("Before diastolic pressure filter...")
    print(df)
    # filter: diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
    diastolic_mask = df["ap_lo"] <= df["ap_hi"]
    # filter: height is less than the 2.5th percentile (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
    low_height_mask = df["height"] >= df["height"].quantile(0.025)
    # filter: height is more than the 97.5th percentile
    high_height_mask = df["height"] <= df["height"].quantile(0.975)
    # filter: weight is less than the 2.5th percentile
    low_weight_mask = df["weight"] >= df["weight"].quantile(0.025)
    # filter: weight is more than the 97.5th percentile
    high_weight_mask = df["weight"] <= df["weight"].quantile(0.975)

    return df[
        diastolic_mask
        & low_height_mask
        & high_height_mask
        & low_weight_mask
        & high_weight_mask
    ]


def save_fig(fig: Union[sns.FacetGrid, plt.figure.Figure], filename: str):
    import os
    from datetime import datetime

    output_dir = "../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"  # Save the plot to a file
    # Save the plot to a file
    fig.savefig(os.path.join(output_dir, filename))


def plot_figure_1(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    df_long = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "active",
            "alco",
            "high_cholesterol",
            "high_gluc",
            "overweight",
            "smoke",
        ],
        ignore_index=False,
    )
    print("Long DF:")
    print(df_long)
    fig = sns.catplot(
        data=df_long,
        kind="count",
        x="variable",
        hue="value",
        col="cardio",
        errorbar="sd",
        palette="dark",
        alpha=0.6,
        height=6,
    )
    fig.despine(left=True)
    fig.set_axis_labels("", "Total")
    fig.set_titles("Cardio Disease: {col_name}")
    fig.legend.set_title("Indicator")

    # Set the title for the entire figure
    plt.pyplot.subplots_adjust(top=0.85)
    fig.fig.suptitle("Indicators by Presence of Cardiovascular Disease")

    save_fig(fig, "indicators")
    plt.pyplot.clf()


def plot_figure_2(df: pd.DataFrame) -> None:
    # Create a figure and axis
    fig, ax = plt.pyplot.subplots(figsize=(16, 8))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    g = sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax, mask=mask
    )
    g.set_title("Correlation Matrix Heatmap")
    save_fig(fig, "correlation")
    plt.pyplot.clf()


if __name__ == "__main__":
    df = load_medical_data()
    df = clean_medical_data(df)

    print("########### Cross Tab")
    print(pd.crosstab(df["cardio"], df["overweight"], margins=True, normalize=False))
    print(pd.crosstab(df["cardio"], df["overweight"], margins=True, normalize=True))
    print("########### Cross Tab - High Cholesterol")
    print(
        pd.crosstab(df["cardio"], df["high_cholesterol"], margins=True, normalize=False)
    )
    print(
        pd.crosstab(df["cardio"], df["high_cholesterol"], margins=True, normalize=True)
    )
    print("########### Cross Tab - High Glucose")
    print(pd.crosstab(df["cardio"], df["high_gluc"], margins=True, normalize=False))
    print(pd.crosstab(df["cardio"], df["high_gluc"], margins=True, normalize=True))

    print("####### Group by")
    print(df.groupby(by="cardio")["overweight"].describe())

    plot_figure_1(df)
    plot_figure_2(df)
