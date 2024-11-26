# Random Forest implementation using RandomForestClassifier from scikit-learn library
# Telco Churn prediction
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


def load_dataset(filename: str) -> pd.DataFrame:
    dtype = {
        "State": "category",
        "Account length": "int16",
        "Area code": "int16",
        "International plan": "category",
        "Voice mail plan": "category",
        "Number vmail messages": "int8",
        "Total day minutes": "float16",
        "Total day calls": "int16",
        "Total day charge": "float16",
        "Total eve minutes": "float16",
        "Total eve calls": "int16",
        "Total eve charge": "float16",
        "Total night minutes": "float16",
        "Total night calls": "int16",
        "Total night charge": "float32",
        "Total intl minutes": "float32",
        "Total intl calls": "int16",
        "Total intl charge": "float32",
        "Customer service calls": "int8",
        "Churn": "bool",
    }
    df = pd.read_csv(f"../../../data/{filename}", dtype=dtype)
    for col in ["International plan", "Voice mail plan"]:
        df[col] = df[col].map({"yes": True, "no": False}).astype(np.bool)
    return df


def simulate_random_forest(x: pd.DataFrame, y: pd.Series) -> None:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(), x.select_dtypes(include="category").columns),
        ],
        remainder="passthrough",
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=1000, class_weight="balanced"),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    df = load_dataset("telecom_churn.csv")
    print(df.info())
    print(df.describe())
    print(df["Churn"].value_counts(normalize=True))
    # Unbalanced data!!!
    x = df.drop(columns=["Churn"])
    y = df["Churn"]
    simulate_random_forest(x, y)
