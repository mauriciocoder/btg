# This script is a decision tree implementation for the medical data. The goal is to create a model to identify whether
# a person has cardiovascular disease or not (Classifier).
# Data set extracted from: https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/demographic-data-analyzer

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


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
    df = pd.read_csv("../../../data/medical_examination.csv", dtype=dtype_mapping)
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


def save_fig(fig: plt.Figure, filename: str):
    import os
    from datetime import datetime

    output_dir = "../../../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"  # Save the plot to a file
    # Save the plot to a file
    fig.savefig(os.path.join(output_dir, filename), dpi=300)


def plot_decision_tree(
    tree: DecisionTreeClassifier, feature_names: list[str], filename: str
):
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_tree(
        tree,
        filled=True,
        feature_names=feature_names,
        rounded=True,
        ax=ax,
    )
    save_fig(fig, filename)


if __name__ == "__main__":
    df = load_medical_data()
    df = clean_medical_data(df)

    x_full = df.drop(columns=["cardio"])
    # The filtered x was based on exploratory data analysis I made
    x_filtered = df[
        [
            "age",
            "weight",
            "ap_lo",
            "cholesterol",
            "gluc",
        ]
    ]
    y = df["cardio"]
    for x in [x_full, x_filtered]:
        # Split the data into 80% training and 20% test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        # Entropy based model
        model = DecisionTreeClassifier(
            criterion="entropy", max_depth=5, min_samples_leaf=100, random_state=42
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        # Notice the accuracy was higher using the full dataset. Scikit-learn was able to find the best
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        plot_decision_tree(model, x.columns, "medical_data_decision_tree")
