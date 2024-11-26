# This script showcases techniques to solve a ml-problem using ensemble (bagging-aggregation).
# The goal is to create a model to predict whether a person has heart disease or not.
# Dataset (Not a competition): https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
# After performing some exploratory data analysis, I created 3 models:
# 1 - A Single Decision Tree model with some hyperparameter tunning;
# 2 - A manual bagging solution, in which I created a loop that bootstraps the data and the features, and perform the classification based on the most frequent class;
# 3 - A bagging solution using the scikit-learn library (BaggingClassifier). In which I use a decision tree as a base estimator. (Which is also considered a Random Forest)
# -> The model with the best results (accuracy) was the bagging solution using the scikit-learn library.
# In a next script (random forest) (src/ml/random_forest/heart.py), I will show how to implement a random forest model.

import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from scipy import stats


def load_dataset(filename: str) -> pd.DataFrame:
    dtype = {
        "Age": "Int8",
        "Sex": "category",
        "ChestPainType": "category",
        "RestingBP": "Int16",
        "Cholesterol": "Int16",
        "FastingBS": "category",
        "RestingECG": "category",
        "MaxHR": "Int16",
        "ExerciseAngina": "category",
        "Oldpeak": "float16",
        "ST_Slope": "category",
        "HeartDisease": "category",
    }
    df = pd.read_csv(f"../../../data/{filename}", dtype=dtype)
    return df


def exploratory_data_analysis(df: pd.DataFrame) -> None:
    print(df.info())
    print(df.describe())
    print(df["HeartDisease"].value_counts())
    print(df["Sex"].value_counts())

    # The countplot, shows the number of samples in each category
    categorical_features = [
        "Sex",
        "ChestPainType",
        "FastingBS",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope",
    ]
    for feature in categorical_features:
        fig, axes = plt.subplots(figsize=(5, 5))
        sns.countplot(data=df, x=feature, hue="HeartDisease", axes=axes)
        print(f"Crosstab between {feature} and HeartDisease")
        print(pd.crosstab(df["HeartDisease"], df[feature], margins=True))

    # The boxplot, shows how the target variable is related to each feature
    numeric_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    for feature in numeric_features:
        fig, axes = plt.subplots(figsize=(5, 5))
        sns.boxplot(data=df, x=feature, y="HeartDisease", ax=axes, showfliers=False)

    # The correlation matrix, shows which features are correlated between each other, indicating candidates for exclusion
    fig, ax = plt.subplots(figsize=(16, 8))
    corr_matrix = df[numeric_features + ["HeartDisease"]].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    g = sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax, mask=mask
    )
    g.set_title("Correlation Matrix Heatmap")
    # plt.show()


def manual_bagging_simulation(
    x: pd.DataFrame, y: pd.DataFrame, bag_size: int = 10
) -> None:
    x_train_original, x_test_original, y_train_original, y_test_original = (
        train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
    )
    print("Starting manual bagging...")
    models = []
    for _ in range(bag_size):
        # Bootstrap sampling: sample rows with replacement
        bootstrap_indices = x_train_original.sample(
            n=len(x_train_original), replace=True
        ).index
        x_bootstrap = x_train_original.loc[bootstrap_indices]
        y_bootstrap = y_train_original.loc[bootstrap_indices]
        # Bootstrap sampling: sample columns with replacement
        cols_size = int(x_train_original.shape[1] / 2)
        x_bootstrap = x_bootstrap.sample(n=cols_size, replace=False, axis=1)
        # Train test split for new samples
        x_train, x_test, y_train, y_test = train_test_split(
            x_bootstrap,
            y_bootstrap,
            test_size=0.3,
            stratify=y_bootstrap,
            random_state=42,
        )
        grid = load_decision_tree_grid(x_train)
        grid.fit(x_train, y_train)
        model = grid.best_estimator_
        accuracy = model.score(x_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        models.append({"model": model, "features": x_bootstrap.columns.tolist()})
    # Testing
    y_preds = []
    for model in models:
        y_pred = model["model"].predict(x_test_original[model["features"]])
        y_preds.append(y_pred)
    y_pred_final = stats.mode(np.array(y_preds, dtype=np.int8), axis=0).mode
    final_accuracy = accuracy_score(
        np.array(y_test_original, dtype=np.int8), y_pred_final
    )
    print("########## Manual Bagging (Random Forest) Classifier Model")
    print(f"Accuracy: {accuracy:.2f}")
    print("##########")


def sklearn_bagging_simulation(x: pd.DataFrame, y: pd.DataFrame):
    pipeline = load_decision_tree_pipeline(x)
    pipeline.set_params(
        classifier=BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=200,
            bootstrap=True,
            bootstrap_features=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )
    )
    # We do not set param grid (max_depth, min_samples_leaf) here because we want to force the overfitting
    # The aggregation will handle any overfitting that may exist
    grid = GridSearchCV(pipeline, {}, cv=5, scoring="accuracy", n_jobs=-1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=42
    )
    grid.fit(x_train, y_train)
    y_predict = grid.predict(x_test)
    accuracy = accuracy_score(y_predict, y_test)
    print("########## SKLearn Bagging (Random Forest) Classifier Model")
    print(f"Accuracy: {accuracy:.2f}")
    print("##########")


def baseline_simulation(x: pd.DataFrame, y: pd.DataFrame):
    # Based on results from exploratory data analysis, I'll create a baseline model
    # Inspecting the Crosstab between ST_Slope and HeartDisease, we can create the model based on it
    conditions = [
        x["ST_Slope"] == "Down",
        x["ST_Slope"] == "Flat",
        x["ST_Slope"] == "Up",
    ]
    choices = [1, 1, 0]
    y_predict = np.select(conditions, choices, default=0)
    # Calculate accuracy
    accuracy = accuracy_score(y.cat.codes, y_predict)
    print("########## Baseline Model")
    print(f"Accuracy: {accuracy:.2f}")
    print("##########")


def single_classifier_simulation(x: pd.DataFrame, y: pd.DataFrame):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=42
    )
    grid_search = load_decision_tree_grid(x)
    grid_search.fit(x_train, y_train)
    print("Best parameters: ", grid_search.best_params_)

    model = grid_search.best_estimator_
    accuracy = model.score(x_test, y_test)
    print("########## Single Classifier Model")
    print(f"Accuracy: {accuracy:.2f}")
    print("##########")


def load_decision_tree_grid(x: pd.DataFrame):
    pipeline = load_decision_tree_pipeline(x)
    # Define parameter grid for GridSearchCV
    param_grid = {
        "classifier__max_depth": [3, 4, 5, 6, 8, 10, 20, None],  # Depth of the tree
        "classifier__min_samples_split": [
            2,
            5,
            10,
            20,
            50,
            100,
        ],  # Min samples to split
        "classifier__min_samples_leaf": [
            1,
            2,
            5,
            10,
            20,
            50,
            100,
            200,
        ],  # Min samples in a leaf
        "classifier__criterion": ["gini", "entropy"],  # Splitting criterion
    }
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    return grid_search


def load_decision_tree_pipeline(x):
    cat_features = x.select_dtypes(include="category").columns
    num_features = x.select_dtypes(exclude="category").columns
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(),
                cat_features,
            ),  # Apply LabelEncoder to categorical columns
            (
                "num",
                MinMaxScaler(),
                num_features,
            ),  # Apply StandardScaler to numerical columns
        ],
        remainder="passthrough",  # Leave other columns unchanged (in case of non-categorical columns)
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )
    return pipeline


if __name__ == "__main__":
    df = load_dataset("heart.csv")
    # exploratory_data_analysis(df)
    x = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    baseline_simulation(x, y)
    single_classifier_simulation(x, y)
    # manual_bagging_simulation(x, y)
    sklearn_bagging_simulation(x, y)


"""
######## Data Dictionary:

Attribute Information
Age: age of the patient [years]
Sex: sex of the patient [M: Male, F: Female]
ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
RestingBP: resting blood pressure [mm Hg]
Cholesterol: serum cholesterol [mm/dl]
FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
Oldpeak: oldpeak = ST [Numeric value measured in depression]
ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
HeartDisease: output class [1: heart disease, 0: Normal]


######## Exploratory Data Analysis:

### General Info
HeartDisease
1    508
0    410
Name: count, dtype: int64

Sex
M    725
F    193
Name: count, dtype: int64


### From countplots of categorical features
From Sex distribution:
1) Approximately one third of women in the dataset has heart disease;
2) The majority of men in the dataset has heart disease;
--- Conclusion ---
Men is more propense to have heart disease


From Chest Pain Type:
1) Out of people with AYS (Asymptomatic), ~400 have heart disease, ~100 have no heart disease;
2) Out of people with ATA, ~25 have heart disease, ~150 have no heart disease; 
3) People with TA is almost equally distributed;
4) More people with NAP does not have a heart disease;
--- Conclusion ---
AYS (Asymptomatic) is a value that could be a good classifier for identifying heart disease


From FastingBS:
1) People with fasting blood sugar > 120 mg/dl has 77% change of having heart disease;
2) People with low fasting blood sugar are almost equally distributed;
--- Conclusion ---
Someone having high FastingBS, is very propense (77% of chance) to have hert disease; 


From RestingECG:
1) People having ST-T wave abnormality (ST) has ~50% more chances of having a heart disease
2) Other values are equally distributed
--- Conclusion ---
ST is a value that could be a good classifier for identifying heart disease


From ExerciseAngina (Chest pain while working out):
1) People having ExerciseAngina has ~80% chance of having heart disease;
2) People with no ExerciseAngina has ~66% chance of not having heart disease;
--- Conclusion ---
People having ExerciseAngina is a good classifier for identifying heart disease


From ST_Slope (rate of change between ventricular polarization during electrocardiogram (ECG))
1) People with Down and Flat are much more propense of having heart disaease than people with UP
--- Conclusion ---
This is a highly correlated feature

### From correlation matrix of numerical features
1) All numeric features are not correlated with each other;
2) The "HeartDisease" category has a |0.4| correlation with "MaxHR" and "Oldpeak";
"""
