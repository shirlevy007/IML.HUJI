from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import os
import plotly.express as px
from math import atan2, pi
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


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(os.path.join(PATH, f))

        # Fit models and predict over training set
        lda = LDA()
        lda = lda.fit(X, y)
        lda_pred = lda.predict(X)
        # quit()
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        symbols = np.array(["circle", "x", "diamond"])
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=
                            # ["gaussian naive bayes model with accuracy = " + str(accuracy(y, predict_naive_bayes)),
                             ["linear discriminant analysis model with accuracy = " + str(accuracy(y, lda_pred))],
                            horizontal_spacing=0.01, vertical_spacing=.03)


        # Add traces for data-points setting symbols and colors
        fig.add_traces(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                  marker=dict(color=lda_pred.astype(int),
                                              symbol=symbols[y.astype(int)])), rows=1, cols=2)
        fig.update_layout(title=f,
                          margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        # fig.add_traces(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
        #                           marker=dict(color=predict_naive_bayes.astype(int),
        #                                       symbol=symbols[y.astype(int)])), rows=1, cols=1)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers", showlegend=False,
                                 marker=dict(size=9, color="black", symbol="x")), row=1, col=2)
        # fig.add_trace(go.Scatter(x=naive_bayes_model.mu_[:, 0], y=naive_bayes_model.mu_[:, 1], mode="markers",
        #                          showlegend=False, marker=dict(size=9, color="black", symbol="x")), row=1, col=1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(lda.mu_)):
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)
            # fig.add_trace(get_ellipse(naive_bayes_model.mu_[i], np.diag(naive_bayes_model.vars_[i])), row=1, col=1)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
