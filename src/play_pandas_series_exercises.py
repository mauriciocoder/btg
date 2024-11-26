import pandas as pd
import numpy as np

if __name__ == "__main__":
    print("Create an empty pandas Series")
    s = pd.Series()
    print(s)

    print("Given the X python list convert it to an Y pandas Series")
    x = ["Mauricio", "Sara"]
    s = pd.Series(x)
    print(s)

    print("Given the X pandas Series, name it 'My letters'")
    s.name = "My letters"
    print(s)

    print("Given the X pandas Series, show its values")
    print(s.values)
    print(type(s.values))

    print("Assign index names to the given X pandas Series")
    s = pd.Series([55, 65, 85], dtype=np.int8)
    s.index = ["Sara", "Eurice", "Mauricio"]
    s.name = "Weight"
    print(s)

    print("Given the X pandas Series, show its first element")
    print(s.iloc[0])

    print("Given the X pandas Series, show its last element")
    print(s.iloc[-1])

    print("Given the X pandas Series, show all middle elements")
    print(s.iloc[1:-1])

    print("Given the X pandas Series, show the elements in reverse position")
    print(s.iloc[::-1])

    print("Given the X pandas Series, show the first and last elements")
    print(s.iloc[[0, -1]])

    print("Convert the given integer pandas Series to float")
    s = pd.Series([1, 2, 3], dtype=np.int8)
    s = s.astype(np.float64)
    print(s)

    print("Reverse the given pandas Series (first element becomes last)")
    s = pd.Series([1, 2, 3], index=["A", "B", "C"], dtype=np.int8)
    s = s[::-1]
    print(s)

    print("Order (sort) the given pandas Series")
    s = pd.Series([1, 3, 2], index=["A", "B", "C"], dtype=np.int8)
    s = s.sort_values(ascending=True)
    print(s)

    print("Given the X pandas Series, set the fifth element equal to 10")
    s = pd.Series(
        [1, 3, 2, 5, 6, 7], index=["A", "B", "C", "D", "E", "F"], dtype=np.int8
    )
    s.iloc[4] = 10
    print(s)

    print("Given the X pandas Series, change all the middle elements to 0")
    s = pd.Series(
        [1, 3, 2, 5, 6, 7], index=["A", "B", "C", "D", "E", "F"], dtype=np.int8
    )
    s.iloc[1:-1] = 0
    print(s)

    print("Given the X pandas Series, add 5 to every element")
    s = pd.Series(
        [1, 3, 2, 5, 6, 7], index=["A", "B", "C", "D", "E", "F"], dtype=np.int8
    )
    s += 5
    print(s)

    print("Given the X pandas Series, make a mask showing negative elements")
    print("Given the X pandas Series, get the negative elements")
    s = pd.Series(
        [-1, 3, -2, 5, -6, 7], index=["A", "B", "C", "D", "E", "F"], dtype=np.int8
    )
    mask = s < 0
    print(s[mask])

    print("Given the X pandas Series, get numbers higher than 5")
    s = pd.Series(
        [-1, 3, -2, 5, -6, 7], index=["A", "B", "C", "D", "E", "F"], dtype=np.int8
    )
    mask = s > 5
    print(s[mask])

    print("Given the X pandas Series, get numbers higher than the elements mean")
    s = pd.Series(
        [-1, 3, -2, 5, -6, 7], index=["A", "B", "C", "D", "E", "F"], dtype=np.int8
    )
    mask = s > s.mean()
    print(s[mask])

    print("Given the X pandas Series, get numbers equal to 2 or 10")
    s = pd.Series(
        [-1, 2, -2, 10, -6, 7], index=["A", "B", "C", "D", "E", "F"], dtype=np.int8
    )
    mask = (s == 2) | (s == 10)
    print(s[mask])

    print("Given the X pandas Series, return True if none of its elements is zero")
    s = pd.Series(
        [-1, 2, -2, 1, -6, 7], index=["A", "B", "C", "D", "E", "F"], dtype=np.int8
    )
    print((s != 0).all())

    print("Given the X pandas Series, return True if any of its elements is zero")
    print((s == 0).any())

    print("Given the X pandas Series, show the sum of its elements")
    print(s.sum())

    print("Given the X pandas Series, show the mean value of its elements")
    print(f"mean={s.mean():.2f}")

    print("Given the X pandas Series, show the max value of its elements")
    print(f"max={s.max()}")
