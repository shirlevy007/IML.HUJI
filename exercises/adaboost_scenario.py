import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaBoost = AdaBoost(DecisionStump, n_learners)
    adaBoost.fit(train_X, train_y)

    train_error, test_error = [], []

    for t in range(1, n_learners):
        train_error.append(adaBoost.partial_loss(train_X, train_y, t))
        test_error.append(adaBoost.partial_loss(test_X, test_y, t))

    figQ1 = go.Figure()
    figQ1.add_trace(go.Scatter(x=np.array(range(1, n_learners)), y=train_error, mode="lines", name='train'))
    figQ1.add_trace(go.Scatter(x=np.array(range(1, n_learners)), y=test_error, mode="lines", name='test'))
    figQ1.update_layout(title="ADABOOST the training and test errors as a function of the number of fitted learners ",
                        xaxis_title="Learners", yaxis_title="Adaboost error")
    figQ1.show()

    symbols = np.array(["circle", "x"])
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    figQ2 = make_subplots(rows=2, cols=2,
                          subplot_titles=[rf'{i} fitted learners' for i in T],
                          horizontal_spacing=0.01, vertical_spacing=0.05)
    for i, t in enumerate(T):
        figQ2.add_traces([decision_surface(lambda x: adaBoost.partial_predict(x, t), lims[0], lims[1], showscale=False),
                          go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                     marker=dict(color=test_y.astype(int), symbol=symbols, colorscale=[custom[0], custom[-1]],
                                                 line=dict(color="black", width=1)))],
                         rows=(i // 2) + 1, cols=(i % 2) + 1)

    figQ2.update_layout(title=rf'Decision boundary obtained by using the the ensemble up to iteration 5, 50, 100 and 250',
                        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)

    figQ2.show()

    # Question 3: Decision surface of best performing ensemble
    losses = []
    for t in range(1, n_learners):
        losses.append(adaBoost.partial_loss(test_X, test_y, t))
    min_ind = np.argmin(losses) + 1
    y_pred = adaBoost.partial_predict(test_X, min_ind)

    figQ3 = go.Figure()
    figQ3.add_traces([decision_surface(lambda x: adaBoost.partial_predict(x, min_ind), lims[0], lims[1]),
                      go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                 marker=dict(color=test_y.astype(int), symbol=symbols, colorscale=[custom[0], custom[-1]],
                                             line=dict(color="black", width=1)))])
    figQ3.update_layout(
        title=rf'Decision surface of ensemble that achieves lowest test error, size {min_ind}.'
              rf'Accuracy is {accuracy(test_y, y_pred)}',
        margin=dict(t=100))

    figQ3.show()

    # Question 4: Decision surface with weighted samples
    figQ4 = go.Figure()
    figQ4.add_traces([decision_surface(adaBoost.predict, lims[0], lims[1]),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                                 marker=dict(color=train_y.astype(int), symbol=symbols, colorscale=[custom[0], custom[-1]],
                                             line=dict(color="white", width=1),
                                             size=adaBoost.D_/np.max(adaBoost.D_)*10))])
    figQ4.update_layout(
        title=rf'Decision surface of ensemble with weighted samples',
        margin=dict(t=100))

    figQ4.show()

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0.0)
    fit_and_evaluate_adaboost(0.4)
