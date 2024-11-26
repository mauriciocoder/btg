# This script is a continuation of the previous script (decision tree) (src/ml/bagging/heart.py).
# The goal is to create a model to predict whether a person has heart disease or not.
# Dataset (Not a competition): https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
# I created a model using RandomForest (Random Forest) with some hyperparameter tunning.
# The results observed were very similar to the ones observed in the simulation BaggingClassifier (src/ml/bagging/heart.py).
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pandas.core.common import random_state
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
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


def sklearn_random_forest_simulation(x: pd.DataFrame, y: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=42
    )
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
        remainder="passthrough",  # Leave other columns unchanged
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
            ),
        ]
    )
    param_grid = {
        "classifier__criterion": ["gini", "entropy", "log_loss"],  # Splitting criterion
        "classifier__max_features": [
            "sqrt",
            "log2",
            None,
        ],  # Number of features to consider for splitting
    }
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)
    y_pred = grid.predict(x_test)
    print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    df = load_dataset("heart.csv")
    # Exploratory data analysis was done in the previous script
    x = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"].astype("int")
    sklearn_random_forest_simulation(x, y)


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
