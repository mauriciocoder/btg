# Taken from https://mlcourse.ai/book/topic04/topic4_linear_models_part4_good_bad_logit_movie_reviews_XOR.html
# The goal is to implement a classifier for the XOR operation. If you look at the scatterplot generated,
# you will see that we can't easily evaluate what is the decision curve. So I decided to
# solve it by using the LWLR (Locally Weighted Logistic Regression) algorithm. The authors are suggesting to use
# quadratic features, which may capture the linearity (squared) of the data.
# Remember that LWLR is a lazy learner, so the algorithm will train a new model for each test point.
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV

# This is the gridsearch which will encapsulate the pipeline and parameterize the model
from sklearn.model_selection import train_test_split


def plot_scatter(x: np.ndarray, y: np.ndarray) -> None:
    fig, axes = plt.subplots(figsize=(10, 10))
    axes.scatter(x[:, 0], x[:, 1], s=30, c=y, cmap=plt.cm.Paired)


def load_weights(
    x_train: np.ndarray, x_test_i: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    # x_train.shape = (m, n), where m = number of training examples, n = number of features
    # x_test_i.shape = (1, n)
    dists = np.power(x_test_i - x_train, 2).sum(axis=1)
    # dists.shape = (m, 1), each point has a distance to x_test_i
    # Weights are based on Gaussian formula
    # weights.shape = (m, 1). each point has a corresponding weight
    return np.exp(-np.power(dists, 2) / (2 * np.power(tau, 2)))


if __name__ == "__main__":
    # Pseudo ramdom data
    x_0 = np.random.rand(600, 1) * 2
    x_1 = np.random.rand(600, 1) * 2
    y = np.logical_xor(x_0 > 1, x_1 > 1)
    x = np.concatenate([x_0, x_1], axis=1)
    x_scaled = MinMaxScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3)
    plot_scatter(x_train, y_train)
    plt.show()

    y_predict = []
    for x_test_i in x_test:
        weights = load_weights(x_train, x_test_i, 0.0005)
        model = LogisticRegressionCV(cv=5, max_iter=10000)
        result = model.fit(x_train, y_train.flatten(), sample_weight=weights).predict(
            [x_test_i]
        )
        y_predict.append(result)

    acc = accuracy_score(y_test, y_predict)
    print(f"Accuracy for LWR: {acc}")
