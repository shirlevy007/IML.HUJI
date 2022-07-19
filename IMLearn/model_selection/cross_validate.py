from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_scores: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    train_scores, val_scores = [], []
    partitions = np.mod(np.arange(len(X)), cv)
    # X_folds = np.array_split(X, cv)
    # y_folds = np.arraay_split(y, cv)
    for i in range(cv):
        X_val, y_val = X[partitions == i], y[partitions == i] #choose different cal each time
        X_train, y_train = X[partitions != i], y[partitions != i] #all the rest := train

        estimator.fit(X_train, y_train)
        y_pred_train = estimator.predict(X_train)
        y_pred_val = estimator.predict(X_val)

        t_score = scoring(y_train, y_pred_train)
        train_scores.append(t_score)
        v_score = scoring(y_val, y_pred_val)
        val_scores.append(v_score)

    return np.average(train_scores), np.average(val_scores)


