# https://mlcourse.ai/book/topic03/assignment03_decision_trees.html#assignment03
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


def save_fig(fig: plt.Figure, filename: str):
    output_dir = "../../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"
    fig.savefig(os.path.join(output_dir, filename))


def plot_decision_tree(
    tree: DecisionTreeClassifier, feature_names: list[str], filename: str
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(15, 15))
    plot_tree(
        tree,
        filled=True,
        feature_names=feature_names,
        rounded=True,
        ax=ax,
    )
    save_fig(fig, filename)
    return fig, ax


def intersect_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame]:
    common_feat = list(set(train.keys()) & set(test.keys()))
    return train[common_feat], test[common_feat]


def encode_to_onehot_df(d: dict[str, list[str]], target: str = None) -> pd.DataFrame:
    features = list(d.keys())
    if target:
        features.remove(target)
    return pd.get_dummies(pd.DataFrame(d), columns=features)


def load_test_data() -> pd.DataFrame:
    df_test = {
        "Looks": ["handsome", "handsome", "repulsive"],
        "Alcoholic_beverage": ["no", "yes", "yes"],
        "Eloquence": ["average", "high", "average"],
        "Money_spent": ["lots", "little", "lots"],
    }
    return encode_to_onehot_df(df_test)


def load_training_data() -> pd.DataFrame:
    df_train = {
        "Looks": [
            "handsome",
            "handsome",
            "handsome",
            "repulsive",
            "repulsive",
            "repulsive",
            "handsome",
        ],
        "Alcoholic_beverage": ["yes", "yes", "no", "no", "yes", "yes", "yes"],
        "Eloquence": [
            "high",
            "low",
            "average",
            "average",
            "low",
            "high",
            "average",
        ],
        "Money_spent": [
            "lots",
            "little",
            "lots",
            "little",
            "lots",
            "lots",
            "lots",
        ],
        "Will_go": [1, 0, 1, 0, 0, 1, 1],
    }
    return encode_to_onehot_df(df_train, "Will_go")


def get_entropy(system: pd.Series) -> float:
    p = system.value_counts(normalize=True)
    return np.sum(p * np.log2(1.0 / p))


def get_info_gain(entropy_zero: float, systems: list[pd.Series]) -> float:
    total_items = sum(len(system) for system in systems)
    entropy_mean = 0
    for system in systems:
        system_len = len(system)
        entropy_mean += (system_len / total_items) * get_entropy(system)
    return entropy_zero - entropy_mean


if __name__ == "__main__":
    df_train = load_training_data()
    y = df_train["Will_go"]
    initial_entropy = get_entropy(y)
    print(f"Initial Entropy: {initial_entropy}")
    y1 = df_train[df_train["Looks_handsome"] == 1]["Will_go"]
    print(f"Entropy with Looks_handsome == 1: {get_entropy(y1)}")
    y0 = df_train[df_train["Looks_handsome"] == 0]["Will_go"]
    print(f"Entropy with Looks_handsome == 0: {get_entropy(y0)}")
    print(f"Information gain: {get_info_gain(initial_entropy, [y1, y0])}")

    df_train, df_test = intersect_features(train=df_train, test=load_test_data())
    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(df_train, y)
    plot_decision_tree(model, df_train.columns, "decision_tree_plot_will_go")

    y_predict = model.predict(df_test)
