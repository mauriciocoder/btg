# https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/demographic-data-analyzer
import pandas as pd
import numpy as np


def load_demographic_data() -> pd.DataFrame:
    df = pd.read_csv("../../data/adult.data.csv")
    return df


if __name__ == "__main__":
    df = load_demographic_data()
    print(df)
    print(df.head())
    print(df.describe())

    print(
        "How many people of each race are represented in this dataset? This should be a Pandas series with race names as the index labels. (race column)"
    )
    x = df["race"].value_counts().astype(np.int32)
    print(x)

    print("What is the average age of men?")
    print(df.loc[df["sex"] == "Male", "age"].mean())

    print("What is the percentage of people who have a Bachelor's degree?")
    bachelors = df["education"].value_counts()["Bachelors"]
    all = df.shape[0]
    perc = round((bachelors / all) * 100, 2)
    print(perc)

    print(
        "What percentage of people with advanced education (Bachelors, Masters, or Doctorate) make more than 50K?"
    )
    advanced_education = ["Bachelors", "Masters", "Doctorate"]
    x = df.loc[df["education"].isin(advanced_education) & (df["salary"] == ">50K")]
    filtered = x.shape[0]
    all = df.loc[df["education"].isin(advanced_education)].shape[0]
    print(round((filtered / all) * 100, 2))

    print("What percentage of people without advanced education make more than 50K?")
    x = df.loc[~(df["education"].isin(advanced_education)) & (df["salary"] == ">50K")]
    filtered = x.shape[0]
    all = df.loc[~df["education"].isin(advanced_education)].shape[0]
    print(round((filtered / all) * 100, 2))

    print("What is the minimum number of hours a person works per week?")
    x = df["hours-per-week"]
    print(x.min())

    print(
        "What percentage of the people who work the minimum number of hours per week have a salary of more than 50K?"
    )
    min_hour_workers = df.loc[df["hours-per-week"] == x.min()]
    high_salary_min_hour_workers = min_hour_workers.loc[
        min_hour_workers["salary"] == ">50K"
    ]
    print(
        round(
            (high_salary_min_hour_workers.shape[0] / min_hour_workers.shape[0]) * 100, 2
        )
    )

    print(
        "What country has the highest percentage of people that earn >50K and what is that percentage?"
    )
    country_counts = df["native-country"].value_counts()
    percentage_series = pd.Series()
    for country in country_counts.index:
        all = country_counts[country]
        highest_paid_people = df.loc[
            (df["native-country"] == country) & (df["salary"] == ">50K")
        ].shape[0]
        percentage = round((highest_paid_people / all) * 100, 2)
        percentage_series[country] = percentage
    print(percentage_series.sort_values(ascending=False))

    print("Identify the most popular occupation for those who earn >50K in India.")
    highly_paid_indians = df.loc[
        (df["salary"] == ">50K") & (df["native-country"] == "India")
    ]
    popular_occupations = highly_paid_indians["occupation"].value_counts()
    print(popular_occupations)
