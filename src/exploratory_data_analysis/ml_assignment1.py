# https://mlcourse.ai/book/topic01/assignment01_pandas_uci_adult.html#assignment01
import pandas as pd


def load_adult_data() -> pd.DataFrame:
    df = pd.read_csv("../../data/adult.data.csv")
    return df


if __name__ == "__main__":
    df = load_adult_data()

    print(f"Shape = {df.shape}")

    # 1. How many men and women (sex feature) are represented in this dataset?
    print(df["sex"].value_counts())

    # 2. What is the average age (age feature) of women?
    print(df[df["sex"] == "Female"]["age"].mean())

    # 3. What is the percentage of German citizens (native-country feature)?
    print(df["native-country"].value_counts(normalize=True))

    # 4. What are the mean and standard deviation of age for those who earn more than 50K per year (salary feature)?
    high_salary_people = df[df["salary"] == ">50K"]
    age_mean = high_salary_people["age"].mean()
    age_std = high_salary_people["age"].std()
    print(f"High salary people age mean is {age_mean:.2f}, std is {age_std:.2f}")

    # 5. What are the mean and standard deviation of age for those who earn less than 50K per year?
    low_salary_people = df[df["salary"] == "<=50K"]
    age_mean = low_salary_people["age"].mean()
    age_std = low_salary_people["age"].std()
    print(f"Low salary people age mean is {age_mean:.2f}, std is {age_std:.2f}")

    # 6. Is it true that people who earn more than 50K have at least high school education? (education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters or Doctorate feature)
    print(high_salary_people["education"].value_counts())

    # 7. Find the maximum age of men of Amer-Indian-Eskimo race.
    print(df[df["race"] == "Amer-Indian-Eskimo"]["age"].describe())

    # 8. Among whom is the proportion of those who earn a lot (>50K) greater: married or single men (marital-status feature)? Consider as married those who have a marital-status starting with Married (Married-civ-spouse, Married-spouse-absent or Married-AF-spouse), the rest are considered bachelors.
    print(df["marital-status"].value_counts())
    married_mask = (
        (df["marital-status"] == "Married-civ-spouse")
        | (df["marital-status"] == "Married-spouse-absent")
        | (df["marital-status"] == "Married-AF-spouse")
        | (df["marital-status"] == "Married-AF-spouse")
    )
    print("Married men distribution:")
    print(df[married_mask]["salary"].value_counts(normalize=True))
    print("Single men distribution:")
    print(df[~married_mask]["salary"].value_counts(normalize=True))

    # 9. What is the maximum number of hours a person works per week (hours-per-week feature)? How many people work such a number of hours, and what is the percentage of those who earn a lot (>50K) among them?
    print(df["hours-per-week"].describe())
    # 99
    print(df[df["hours-per-week"] == df["hours-per-week"].max()])
    # 85 people
    print(
        df[df["hours-per-week"] == df["hours-per-week"].max()]["salary"].value_counts(
            normalize=True
        )
    )
    # percentage: 0.294118

    # 10. Count the average time of work (hours-per-week) for those who earn a little and a lot (salary) for each country (native-country). What will these be for Japan?
    print("########## Poor people")
    print(
        df[df["salary"] == "<=50K"].groupby("native-country")["hours-per-week"].mean()
    )
    print("########## Rich people")
    print(df[df["salary"] == ">50K"].groupby("native-country")["hours-per-week"].mean())
