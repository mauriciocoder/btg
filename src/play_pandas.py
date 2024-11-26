import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def save_plot(filename: str):
    import os
    from datetime import datetime

    output_dir = "../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"  # Save the plot to a file
    plt.savefig(os.path.join(output_dir, filename))


if __name__ == "__main__":
    print("Create an empty pandas DataFrame")
    df = pd.DataFrame()
    print(df)
    print("Create a marvel_df pandas DataFrame with the given marvel data")
    marvel_data = [
        ["Spider-Man", "male", 1962],
        ["Captain America", "male", 1941],
        ["Wolverine", "male", 1974],
        ["Iron Man", "male", 1963],
        ["Thor", "male", 1963],
        ["Thing", "male", 1961],
        ["Mister Fantastic", "male", 1961],
        ["Hulk", "male", 1962],
        ["Beast", "male", 1963],
        ["Invisible Woman", "female", 1961],
        ["Storm", "female", 1975],
        ["Namor", "male", 1939],
        ["Hawkeye", "male", 1964],
        ["Daredevil", "male", 1964],
        ["Doctor Strange", "male", 1963],
        ["Hank Pym", "male", 1962],
        ["Scarlet Witch", "female", 1964],
        ["Wasp", "female", 1963],
        ["Black Widow", "female", 1964],
        ["Vision", "male", 1968],
    ]
    heroes = pd.DataFrame(marvel_data, columns=["Name", "Gender", "Year"])
    print(heroes)
    print(heroes.info())

    print("Add index names to the marvel_df (use the character name as index)")
    heroes.set_index("Name", inplace=True)
    print(heroes)

    print("Drop 'Namor' and 'Hank Pym' rows")
    heroes.drop(["Namor", "Hank Pym"], inplace=True)
    print(heroes)

    print("Show the first 5 elements on marvel_df")
    print(heroes.iloc[0:5])

    print("Show the last 5 elements on marvel_df")
    print(heroes.iloc[-5:,])

    print("Show just the sex of the first 5 elements on marvel_df")
    print(heroes.iloc[:5,]["Gender"])

    print("Show the first_appearance of all middle elements on marvel_df")
    print(heroes.iloc[1:-1,]["Year"])

    print("Show the first and last elements on marvel_df")
    print(heroes.iloc[[0, -1]])

    print("Modify the first_appearance of 'Vision' to year 1964")
    heroes.loc["Vision", "Year"] = 1964

    print(
        "Add a new column to marvel_df called 'years_since' with the years since first_appearance"
    )
    import datetime as dt

    heroes["Years Since"] = dt.datetime.now().year - heroes["Year"]
    print(heroes)

    print(
        "Given the marvel_df pandas DataFrame, make a mask showing the female characters"
    )
    mask = heroes["Gender"] == "female"
    print(mask)

    print("Given the marvel_df pandas DataFrame, get the male characters")
    print(heroes[heroes["Gender"] == "male"])

    print(
        "Given the marvel_df pandas DataFrame, get the characters with first_appearance after 1970"
    )
    print(heroes[heroes["Year"] > 1970])

    print(
        "Given the marvel_df pandas DataFrame, get the female characters with first_appearance after 1970"
    )
    print(heroes[(heroes["Year"] > 1970) & (heroes["Gender"] == "female")])

    print("Show basic statistics of marvel_df")
    print(heroes.describe())

    print(
        "Given the marvel_df pandas DataFrame, show the mean value of first_appearance"
    )
    print(heroes["Year"].mean())

    print(
        "Given the marvel_df pandas DataFrame, show the min value of first_appearance"
    )
    print(heroes["Year"].min())

    print(
        "Given the marvel_df pandas DataFrame, get the characters with the min value of first_appearance"
    )
    print(heroes[heroes["Year"] == heroes["Year"].min()])

    print("Reset index names of marvel_df")
    heroes.reset_index(inplace=True)
    print(heroes)

    print("Plot the values of first_appearance")
    heroes["Year"].plot()
    save_plot("heroes")
    plt.clf()

    print("Plot a histogram (plot.hist) with values of first_appearance")
    heroes["Year"].plot(kind="hist", bins=50)
    save_plot("heroes_plot")
