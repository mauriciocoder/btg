# https://mlcourse.ai/book/topic03/assignment03_decision_trees.html#part-3-the-adult-dataset
import os
from collections import namedtuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree, DecisionTreeRegressor, DecisionTreeClassifier
import seaborn as sns


def load_adult_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
    df_train = pd.read_csv(DATA_PATH + "adult_train.csv", sep=";")
    df_test = pd.read_csv(DATA_PATH + "adult_test.csv", sep=";")
    return df_train, df_test


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        (df["Target"] == " >50K.")
        | (df["Target"] == " <=50K.")
        | (df["Target"] == " >50K")
        | (df["Target"] == " <=50K")
    ]
    df.loc[(df["Target"] == " <=50K") | (df["Target"] == " <=50K."), "Target"] = 0
    df.loc[(df["Target"] == " >50K") | (df["Target"] == " >50K."), "Target"] = 1

    df["Age"] = df["Age"].astype(np.int8)
    df["Target"] = df["Target"].astype(np.int8)
    df["Education_Num"] = df["Education_Num"].astype(np.int8)
    df["Hours_per_week"] = df["Hours_per_week"].astype(np.int8)
    df["Capital_Gain"] = df["Capital_Gain"].astype(np.int32)
    df["Capital_Loss"] = df["Capital_Loss"].astype(np.int32)

    categorical_columns = [c for c in df.columns if df[c].dtype == "object"]
    for c in categorical_columns:
        df[c].fillna(df[c].mode()[0], inplace=True)
    numerical_columns = [c for c in df.columns if df[c].dtype != "object"]
    for c in numerical_columns:
        df[c].fillna(df[c].median(), inplace=True)

    df = pd.concat(
        [
            df[numerical_columns],
            pd.get_dummies(df[categorical_columns]),
        ],
        axis=1,
    )
    return df


if __name__ == "__main__":
    df_train, df_test = load_adult_data()
    df_train = clean(df_train)
    df_test = clean(df_test)

    y_train = df_train.pop("Target")
    y_test = df_test.pop("Target")
    df_test["Country_ Holand-Netherlands"] = 0

    x_train = df_train
    x_test = df_test[x_train.columns]

    print(
        "6. What is the test set accuracy of a decision tree with maximum tree depth of 3 and random_state = 17?"
    )
    model = DecisionTreeClassifier(max_depth=3, random_state=17)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    print(
        "3.7. What is the test set accuracy of a decision tree with maximum tree depth of 9 and random_state = 17?"
    )
    model = DecisionTreeClassifier(max_depth=9, random_state=17)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
