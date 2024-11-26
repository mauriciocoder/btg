# Alice Kaggle competition: https://www.kaggle.com/competitions/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/overview
# Goal:
#    To detect an intruder in a network environment. It was give sessions (websites visited with the timestamp (up to 10 websites)) from users
#   and these sessions were previously classified as a session from an intruder or not.
# Solution:
#    From data exploration: It was noticed that the intruder usually starts the sessions in certain days of week and in certains time (check plots).
#   The dataset is highly unbalanced, the vast majority of the sessions are from users who are not the intruder.
#    Modeling: Given the large amount of different websites visited by the users, it was adopted a bag of words approach. Each website has a numeric id.
#    Each exisiting session id becomes a feature. Each session is modeled as row in a dataframe having the session_id as index, and sites [1...<last_site_id>] as columns.
#   Also it was included the day_of_week (0-6, Monday-Sunday) and time (0-23) in the dataframe. The model was then trained with logistic regression set with balanced classes (git favours the intruder, given it was highly unbalanced).
#   Knowing that newton method and L2 regularization is more appropriate for large datasets, this was adopted.
#
import os

import matplotlib
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, FunctionTransformer

import pickle
import seaborn as sns

# Required for live plot
# requires:sudo apt-get install python3-pyqt5
# pip install PyQt5
matplotlib.use("Qt5Agg")  # or 'Qt5Agg', 'MacOSX', etc.
# from utils.plot import save_fig, plot_linechart


def load_sites_dict() -> dict:
    """Returns ids to sites dictionary"""
    with open(
        "../../../data/alice/site_dic.pkl", "rb"
    ) as file:  # 'rb' mode is for reading binary files
        data = pickle.load(file)
    reverted_dic = {v: k for k, v in data.items()}
    return reverted_dic


def load_dataset(file_name: str) -> pd.DataFrame:
    dtype = {f"site{i}": "Int32" for i in range(1, 11)}
    parse_dates = [f"time{i}" for i in range(1, 11)]
    df = pd.read_csv(
        f"../../../data/alice/{file_name}",
        parse_dates=parse_dates,
        index_col="session_id",
        dtype=dtype,
    )
    # df_size_in_bytes = df.memory_usage(index=True, deep=True).sum()
    # print(f"DataFrame size: {df_size_in_bytes / (1024 * 1024):.2f} MB")
    dt = df["time1"].dt
    df["day_of_week"] = dt.dayofweek.astype("int8")
    df["start_hour"] = dt.hour.astype("int8")
    # Merge the columns for sites in a single one. The value is a string separated by spaces. Nans are replaced by 0
    df["sites"] = df[[f"site{i}" for i in range(1, 11)]].apply(
        lambda row: " ".join(row.fillna(0).values.astype(str)), axis=1
    )
    # Dropping the columns
    df.drop(columns=[f"time{i}" for i in range(1, 11)], inplace=True)
    df.drop(columns=[f"site{i}" for i in range(1, 11)], inplace=True)
    return df


def plot_histogram(
    df: pd.DataFrame,
    feature: str,
    title_suffix: str = "",
    density: bool = True,
    xlim: tuple = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))  # Customize figure size if needed
    # df[feature].hist(ax=ax, bins=sorted(df[feature].unique()), density=density)
    sns.histplot(
        df[feature],
        kde=False,
        ax=ax,
        stat="density" if density else "count",
        discrete=True,
    )
    ax.set_xlim(xlim[0], xlim[1])
    # Customize the plot
    ax.set_title(f"Histogram of {feature} - {title_suffix}")
    ax.set_xlabel(f"{feature} Values")
    ax.set_ylabel("Frequency")


if __name__ == "__main__":
    """
    # Graphs plotted for Exploratory Data Analysis
    day_of_week_xlim = (0, 6)
    start_hour_xlim = (0, 23)
    plot_histogram(
        df_train[df_train["target"] == 1], "day_of_week", "Alice", xlim=day_of_week_xlim
    )
    plot_histogram(
        df_train[df_train["target"] == 1], "start_hour", "Alice", xlim=start_hour_xlim
    )
    plot_histogram(
        df_train[df_train["target"] == 0],
        "day_of_week",
        "Others",
        xlim=day_of_week_xlim,
    )
    plot_histogram(
        df_train[df_train["target"] == 0], "start_hour", "Others", xlim=start_hour_xlim
    )
    plt.show()
    """
    # Load the saved model
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
    else:
        df_train = load_dataset("train_sessions.csv")
        x = df_train.drop(columns=["target"])
        y = df_train["target"]

        # stratify=y -> This ensures that both the training and testing sets have the same class distribution
        # as the original dataset, preserving the minority class's presence in both sets.
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, stratify=y
        )

        # Define ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "vectorizer",
                    CountVectorizer(),
                    "sites",
                ),
                ("scaler", MinMaxScaler(), ["day_of_week", "start_hour"]),
                (
                    "poly_features_datetime",
                    PolynomialFeatures(degree=1, include_bias=False),
                    ["day_of_week", "start_hour"],
                ),
            ],
            remainder="passthrough",  # Preserve all other columns unchanged
        )
        # Define Pipeline
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    LogisticRegression(
                        solver="newton-cg",
                        penalty="l2",
                        class_weight="balanced",
                        n_jobs=-1,
                        max_iter=500,
                    ),
                ),
            ]
        )
        """
        Encapsulating a pipeline in GridSearchCV is a best practice because it ensures 
        that all preprocessing steps are properly included in the cross-validation process, 
        leading to more reliable model evaluation and tuning.
        """
        # C: The inverse of the regularization strength in models like LogisticRegression.
        # features__degree: Degree of the polynomial features for day of week and starting_hour
        # hyper_params = {}
        hyper_params = {
            "regressor__C": np.logspace(-2, 1, 10),
            "preprocessor__poly_features_datetime__degree": range(3, 8),
        }
        grid = GridSearchCV(pipeline, hyper_params, cv=3, n_jobs=-1)
        grid.fit(x_train, y_train)
        print("Best Params:")
        print(grid.best_params_)
        # Save the best estimator
        model = grid.best_estimator_
        with open("best_model.pkl", "wb") as f:
            pickle.dump(model, f)

        y_train_pred = grid.best_estimator_.predict(x_train)
        print("Training classification report:")
        print(classification_report(y_train, y_train_pred))
        print("Testing classification report:")
        y_test_pred = grid.best_estimator_.predict(x_test)
        print(classification_report(y_test, y_test_pred))

        ### The code below is to check if the preprocessor is considering all the site features + the day_of_week and start_hour
        """
        # Suppose you want to evaluate a single row from x_train
        single_row = x_train.iloc[0:1]  # Get the first row as a DataFrame (ensure it's 2D)
    
        # Access the best estimator (which is the full pipeline with the best hyperparameters)
        best_model = grid.best_estimator_
    
        # Access the preprocessor (ColumnTransformer)
        preproc = best_model.named_steps["preprocessor"]
    
        # Transform the single row using the preprocessor
        transformed_single_row = preproc.transform(single_row)
    
        # Convert the result to a dense format (if it's sparse)
        dense_single_row = transformed_single_row.toarray()
    
        # If needed, convert to a DataFrame to view the features with column names
        feature_names = preproc.transformers_[0][1].get_feature_names_out()
        feature_names = np.append(feature_names, ["day_of_week", "start_hour"])
        single_row_df = pd.DataFrame(dense_single_row, columns=feature_names)
        print(single_row_df)
        """
        ### End of test
    # Kaggle test
    print("Running Kaggle test data...")
    x_kaggle_test = load_dataset("test_sessions.csv")
    y_kaggle_predict = model.predict(x_kaggle_test)
    df_submission = pd.DataFrame(
        data=y_kaggle_predict, index=x_kaggle_test.index, columns=["target"]
    )
    df_submission.to_csv("kaggle_submission.csv")
