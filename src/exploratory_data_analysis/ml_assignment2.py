# https://mlcourse.ai/book/topic02/assignment02_analyzing_cardiovascular_desease_data.html
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Literal


def load_medical_data() -> pd.DataFrame:
    # Men are gender=2
    # 24470 men and 45530 women
    types = {
        "age": np.int16,
        "height": np.int16,
        "weight": np.float64,
        "gender": "object",  # 'object' is used for strings, similar to np.object
        "ap_hi": np.int16,
        "ap_lo": np.int16,
        "cholesterol": np.int8,
        "gluc": np.int8,
        "smoke": "bool",  # 'bool' for boolean values
        "alco": "bool",
        "active": "bool",
        "cardio": "bool",
    }
    df = pd.read_csv("../../data/mlbootcamp5_train.csv", dtype=types)

    rows_before_cleaning = df.shape[0]
    print(f"Rows before cleaning: {rows_before_cleaning}")
    df = df[
        (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
        & (df["ap_lo"] <= df["ap_hi"])
    ]
    rows_after_cleaning = df.shape[0]
    print(f"Rows after cleaning: {rows_after_cleaning}")
    print(
        f"Percentage of cleaned data: {(rows_before_cleaning - rows_after_cleaning) / rows_before_cleaning}"
    )

    df["age"] = round(df["age"] / 365.25).astype(np.int8)
    df["age_months"] = round(df["age"] * 12).astype(np.int16)
    df["high_cholesterol"] = df["cholesterol"] > 1
    df["high_gluc"] = df["gluc"] > 1
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["old_and_high_cholesterol"] = (df["age"] >= 55) & (
        df["high_cholesterol"] == True
    )
    df["old_and_high_bmi"] = (df["age"] >= 55) & (df["bmi"] >= 30.0)
    return df


def save_fig(fig: plt.Figure, filename: str):
    import os
    from datetime import datetime

    output_dir = "../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"  # Save the plot to a file
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


def plot_boxplots(df: pd.DataFrame, columns: list[str]) -> None:
    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(12, 24))
    # Plot a histogram for each column
    for i, col in enumerate(columns):
        # Plotting the boxplot with the custom list of colors
        sns.boxplot(x=col, data=df, ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}")
    # Adjust the distance between plots
    plt.subplots_adjust(hspace=0.6)  # Adjust hspace for vertical spacing
    save_fig(fig, "boxplots")


def plot_corr_matrix(
    df: pd.DataFrame, method: Literal["pearson", "kendall", "spearman"] = "pearson"
) -> None:
    fig, axes = plt.subplots(figsize=(24, 24))
    corr = df.corr(method=method)
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=axes)
    plt.title("Correlation Matrix Heatmap")
    save_fig(fig, f"{method}_corr_matrix")


if __name__ == "__main__":
    df = load_medical_data()

    plot_histograms(
        df,
        [
            "age",
            "gender",
            "height",
            "weight",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "high_cholesterol",
            "high_gluc",
            "bmi",
            "old_and_high_cholesterol",
            "old_and_high_bmi",
        ],
        hue="cardio",
    )
    plot_boxplots(
        df,
        [
            "age",
            "gender",
            "height",
            "weight",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
        ],
    )
    plot_corr_matrix(df)
    plot_corr_matrix(df, method="spearman")

    print(df[["cardio", "age"]].corr())
    print(df[["cardio", "weight"]].corr())
    print(df[["cardio", "high_gluc"]].corr())
    print(df[["cardio", "high_cholesterol"]].corr())
    print(df[["cardio", "old_and_high_bmi"]].corr())
    print(df[["cardio", "old_and_high_cholesterol"]].corr())
    print(df[["cardio", "bmi"]].corr())
    print(
        pd.crosstab(
            df["cardio"], df["old_and_high_cholesterol"], margins=True, normalize=True
        )
    )

    # Question 1.2. (1 point). Who more often report consuming alcohol – men or women?
    print(pd.crosstab(df["gender"], df["alco"], margins=True, normalize=True))
    men_qtd = df[df["gender"] == "2"].shape[0]
    men_alco_qtd = df[(df["gender"] == "2") & (df["alco"] == True)].shape[0]
    print(f"men reporting alco ratio = {men_alco_qtd / men_qtd}")
    women_qtd = df[df["gender"] == "1"].shape[0]
    women_alco_qtd = df[(df["gender"] == "1") & (df["alco"] == True)].shape[0]
    print(f"women reporting alco ratio = {women_alco_qtd / women_qtd}")
    if men_alco_qtd > women_alco_qtd:
        print("Men report drinking alcohol more often")
    else:
        print("Women report drinking alcohol more often")

    # Question 1.3. (1 point). What’s the rounded difference between the percentages of smokers among men and women?
    men_smoker_qtd = df[(df["gender"] == "2") & (df["smoke"] == True)].shape[0]
    men_smoker_perc = men_smoker_qtd / men_qtd
    women_smoker_qtd = df[(df["gender"] == "1") & (df["smoke"] == True)].shape[0]
    women_smoker_perc = women_smoker_qtd / women_qtd
    diff = round(men_smoker_perc - women_smoker_perc, 4)
    print(f"Difference between smokers percentage: {diff}")

    # Question 1.4. (1 point). What’s the rounded difference between median values of age (in months) for non-smokers and smokers? You’ll need to figure out the units of feature age in this dataset.
    smokers = df[df["smoke"] == True]
    non_smokers = df[df["smoke"] == False]
    median_ages_smokers = smokers["age_months"].median()
    median_ages_non_smokers = non_smokers["age_months"].median()
    diff = round(median_ages_smokers - median_ages_non_smokers)
    print(
        f"Difference between median values of age between smokers and non-smokers: {diff}"
    )

    # Question 1.5. (2 points). Calculate fractions of ill people (with CVD) in the two groups of people described in the task. What’s the ratio of these two fractions?
    old_people = df[df["age"] >= 60]
    # Group 1 -> age >= 60 and ap_hi < 120
    group1 = old_people[old_people["ap_hi"] < 120]
    group1_cvd_rate = group1[group1["cardio"] == True].shape[0] / group1.shape[0]
    print(f"group1_cvd_rate = {group1_cvd_rate}")
    # Group 2 -> age >= 60 and ap_hi >= 160 and ap_hi < 180
    group2 = old_people[(old_people["ap_hi"] >= 160) & (old_people["ap_hi"] < 180)]
    group2_cvd_rate = group2[group2["cardio"] == True].shape[0] / group2.shape[0]
    print(f"group2_cvd_rate = {group2_cvd_rate}")
    print(f"Ratio is {group2_cvd_rate / group1_cvd_rate}")

    # Quesiton 1.6
    # Median BMI in the sample is within boundaries of normal values.
    print(
        f"Is BMI within the boundaries of normal values: {18.5 <= df["bmi"].median() <= 25}"
    )
    # Women’s BMI is on average higher then men’s.
    men = df[df["gender"] == "2"]
    women = df[df["gender"] == "1"]
    print(
        f"Women’s BMI is on average higher then men’s: {women["bmi"].mean() > men["bmi"].mean()}"
    )
    # Healthy people have higher median BMI than ill people.
    unhealthy = df[df["cardio"] == True]
    healthy = df[df["cardio"] == False]
    print(
        f"Healthy people have higher median BMI than ill people: {healthy["bmi"].median() > unhealthy["bmi"].median()}"
    )
    # In the segment of healthy and non-drinking men BMI is closer to the norm than in the segment of healthy and non-drinking women
    men_seg_mean = healthy[(healthy["gender"] == "2") & (healthy["alco"] == False)][
        "bmi"
    ].mean()
    women_seg_mean = healthy[(healthy["gender"] == "1") & (healthy["alco"] == False)][
        "bmi"
    ].mean()

    men_seg_diff = 0
    if men_seg_mean > 25.0:
        men_seg_diff = men_seg_mean - 25.0
    elif men_seg_mean < 18.5:
        men_seg_diff = 18.5 - men_seg_mean

    women_seg_diff = 0
    if women_seg_mean > 25.0:
        women_seg_diff = women_seg_mean - 25.0
    elif women_seg_mean < 18.5:
        women_seg_diff = 18.5 - women_seg_mean
    print(
        f"In the segment of healthy and non-drinking men BMI is closer to the norm than in the segment of healthy and non-drinking women: {men_seg_diff < women_seg_diff}"
    )
