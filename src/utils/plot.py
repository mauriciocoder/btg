import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def save_fig(fig: plt.Figure, filename: str):
    import os
    from datetime import datetime

    output_dir = "../../../results/plots"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.png"  # Save the plot to a file
    # Save the plot to a file
    fig.savefig(os.path.join(output_dir, filename), dpi=300)


def plot_linechart(
    df: pd.DataFrame,
    x: str,
    y: str = None,
    plot_title: str = None,
    fig: plt.Figure = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    if not fig:
        fig, axes = plt.subplots(figsize=figsize)
    sns.lineplot(data=df, x=x, y=y)
    fig.suptitle(plot_title)
    return fig


def show_3d_scatter(
    x_1: np.ndarray,
    x_2: np.ndarray,
    labels: np.ndarray,
    model: any = None,
) -> None:
    """
    Generates a scatterplot for the x_1, x_2 and labels. If a trained model is passed
    then it also generates the surface of the prediction function.
    The plot generated is not saved to preserve the 3D rotation feature
    :param x_1:
    :param x_2:
    :param labels:
    :param model:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Create the 3D scatter plot
    ax.scatter(x_1, x_2, labels, s=50, c="b", label="3D Data")
    if model:
        # Generate the surface for the loss function
        x1 = np.linspace(0, 1, 100)
        x2 = np.linspace(0, 1, 100)
        # Generates a meshgrid for both x1 and x2 values
        x1_mesh, x2_mesh = np.meshgrid(x1, x2)
        z = []
        # For each value in the meshgrid, predict a value and add it to the z array
        for [a, b] in zip(np.ravel(x1_mesh), np.ravel(x2_mesh)):
            z.append(model.predict(np.array([[a, b]])).flatten())
        # Reshape the z array to match the shape of the meshgrid
        z = np.array(z).reshape(x1_mesh.shape)
        # Drawing the surface, notice x1_mesh, x2_mesh, and z are all with the same shape
        ax.plot_surface(x1_mesh, x2_mesh, z, alpha=0.7)
        # End of the curve
    # Labels and title
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Data points and prediction curve")
    ax.legend()
    plt.show()
