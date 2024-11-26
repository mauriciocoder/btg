# This script is a k-NN implementation for the medical data. The goal is to create a model to identify whether
# a person has cardiovascular disease or not (Classifier).
# Data set extracted from: https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/demographic-data-analyzer

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
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
    # filter: diastolic pressure is positive
    diastolic_positive_mask = df["ap_lo"] > 0
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
        diastolic_positive_mask
        & diastolic_mask
        & low_height_mask
        & high_height_mask
        & low_weight_mask
        & high_weight_mask
    ]


def save_fig(fig: plt.Figure, filename: str):
    import os
    from datetime import datetime

    output_dir = "../../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"  # Save the plot to a file
    # Save the plot to a file
    fig.savefig(os.path.join(output_dir, filename), dpi=300)


if __name__ == "__main__":
    df = load_medical_data()
    df = clean_medical_data(df)

    print("Distribution:")
    print(df["cardio"].value_counts(normalize=True))

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
    print(x_filtered.describe(include="all"))
    scaler = MinMaxScaler()
    x_normalized = pd.DataFrame(
        scaler.fit_transform(x_filtered), columns=x_filtered.columns
    )
    print(x_normalized.describe(include="all"))

    # Split the data into 80% training and 20% test
    x_train, x_test, y_train, y_test = train_test_split(
        x_normalized, y, test_size=0.2, random_state=42
    )

    # Create a KNN classifier
    knn = KNeighborsClassifier(
        n_neighbors=200, weights="uniform", algorithm="ball_tree"
    )
    knn.fit(x_train, y_train)

    param_grid = {
        "n_neighbors": [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560],
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute"],
    }

    grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, verbose=3)

    # Fit the GridSearchCV object to the data
    grid_search.fit(x_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    y_predicted = grid_search.predict(x_test)
    accuracy_score = accuracy_score(y_test, y_predicted)
    print(f"Accuracy score: {accuracy_score}")
