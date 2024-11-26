# https://mlcourse.ai/book/topic03/assignment03_decision_trees.html#part-2-functions-for-calculating-entropy-and-information-gain
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


def information_gain(entropy_0: float, systems: list[list[int]]) -> float:
    total_items = sum(len(system) for system in systems)
    entropy_mean = 0
    for system in systems:
        entropy_mean += (len(system) / total_items) * entropy(system)
    return entropy_0 - entropy_mean


def entropy(group: list[int]) -> float:
    s = set(group)
    d = {element: group.count(element) for element in s}
    entropy = 0
    for element in s:
        p = d[element] / len(group)
        entropy += p * np.log2(1.0 / p)
    return entropy


if __name__ == "__main__":
    balls = [1 for i in range(9)] + [0 for i in range(11)]
    print(balls)

    # two groups
    balls_left = [1 for i in range(8)] + [0 for i in range(5)]  # 8 blue and 5 yellow
    balls_right = [1 for i in range(1)] + [0 for i in range(6)]  # 1 blue and 6 yellow

    print("3.3. What is the entropy of a state given by a list balls_left?")
    print(entropy(balls_left))

    print(
        "3.4. What is the entropy of a fair dice? (where we look at a dice as a system with 6 equally probable states"
    )
    print(entropy([1, 2, 3, 4, 5, 6]))

    print(
        "3.5. What is the information gain of splitting the initial dataset into balls_left and balls_right?"
    )
    print(information_gain(entropy(balls), [balls_left, balls_right]))
