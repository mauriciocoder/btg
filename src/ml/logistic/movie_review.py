# Taken from https://mlcourse.ai/book/topic04/topic4_linear_models_part4_good_bad_logit_movie_reviews_XOR.html
# The goal is to implement a binary classifier such that it identifies if a review is positive (Good) or negative (Bad).
# The training data has 12500 Good x 12500 Bad. The strategy adopted is to use a bag of words. Every word used is a feature.
# This is also an example of how to create a pipeline in Scikit learn and use a gridSearch.
import numpy as np
from matplotlib import pyplot as plt

"""
The load_files function from sklearn.datasets is used to load datasets where the data is organized in a folder structure,
typically for text classification or other tasks involving documents stored in files.
It is particularly helpful when you have labeled data organized into directories,
with each directory name corresponding to a class label.
"""
from sklearn.datasets import load_files

# The CountVectorizer is used to convert text into a vector of word counts.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# This will create a pipeline that standardizes the data, then runs logistic regression.
from sklearn.pipeline import Pipeline

# This is the gridsearch which will encapsulate the pipeline and parameterize the model
from sklearn.model_selection import GridSearchCV


def plot_coefficients(model, feature_names, n_top_features=25):
    # get coefficients with large absolute values
    coef = model.coef_.ravel()
    pos_coeffs = np.argsort(coef)[-n_top_features:]
    neg_coeffs = np.argsort(coef)[:n_top_features]
    top_coeffs = np.hstack([neg_coeffs, pos_coeffs])
    # plot them
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coeffs]]
    plt.bar(np.arange(2 * n_top_features), coef[top_coeffs], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(
        np.arange(1, 1 + 2 * n_top_features),
        feature_names[top_coeffs],
        rotation=60,
        ha="right",
    )
    plt.show()


def plot_grid_scores(grid, param_name):
    plt.figure(figsize=(15, 5))
    plt.plot(
        grid.param_grid[param_name],
        grid.cv_results_["mean_train_score"],
        color="green",
        label="train",
    )
    plt.plot(
        grid.param_grid[param_name],
        grid.cv_results_["mean_test_score"],
        color="red",
        label="test",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    movie_reviews = load_files(
        "../../../data/aclImdb/train/", categories=["neg", "pos"]
    )  # This will load the files from the data folder, the foler structure is: .../train/neg, .../train/pos and create an object having the data (comments) and the target (0, 1)
    text_train, y_train = movie_reviews.data, movie_reviews.target
    reviews_test = load_files("../../../data/aclImdb/test", categories=["neg", "pos"])
    text_test, y_test = reviews_test.data, reviews_test.target

    # This pipeline will run the following for the fit call:
    # 1. CountVectorizer -> Will transform the text (training and testing objects) into a vector of word counts
    # 2. LogisticRegression -> Will apply the previous output to a logistic regression.
    pipeline = Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("regressor", LogisticRegression(solver="lbfgs", n_jobs=-1)),
        ],
        verbose=True,
    )

    """
    Encapsulating a pipeline in GridSearchCV is a best practice because it ensures 
    that all preprocessing steps are properly included in the cross-validation process, 
    leading to more reliable model evaluation and tuning.
    """
    # C: The inverse of the regularization strength in models like LogisticRegression.
    params = {"regressor__C": np.logspace(-5, 0, 6)}
    grid = GridSearchCV(pipeline, params, return_train_score=True, cv=3, n_jobs=-1)
    grid.fit(text_train, y_train)
    print(f"Train score: {grid.score(text_train, y_train)}")
    print(f"Test score: {grid.score(text_test, y_test)}")

    # Get the best pipeline from GridSearchCV
    best_pipeline = grid.best_estimator_
    # Access the LogisticRegression step
    model = best_pipeline.named_steps["regressor"]
    # Access the CountVectorizer step
    vectorizer = best_pipeline.named_steps["vectorizer"]
    # Retrieve feature names
    feature_names = vectorizer.get_feature_names_out()
    plot_coefficients(model, feature_names)
    plot_grid_scores(grid, "regressor__C")
