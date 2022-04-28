from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import os
import plotly.express as px
import IMLearn.learners.classifiers.perceptron
from plotly.subplots import make_subplots

pio.templates.default = "simple_white"

PATH = "datasets"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    print(f'{filename}')
    dataframe = np.load(filename)
    # dataframe = dataframe.dropna()
    # dataframe = dataframe.drop_duplicates()
    X = dataframe[:, :2]
    y = dataframe[:, 2]
    return X, y


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(os.path.join(PATH, f))

        losses = []

        def callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X, y))
            # print(losses)

        # Fit Perceptron and record loss in each fit iteration
        perceptron = Perceptron(callback=callback)
        perceptron.fit(X, y)

        # Plot figure
        fig = px.line(x=range(len(losses)), y=losses,
                      title="The perceptron algorithm's training loss values as a function of the training iterations",
                      labels={"x": "Training iterations", "y": "Training loss values"})
        fig.show()

def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        raise NotImplementedError()

        # Fit models and predict over training set
        raise NotImplementedError()

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
