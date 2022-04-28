from typing import NoReturn

import pandas
import IMLearn.metrics
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        # self.mu_ = np.ndarray((len(self.classes_), len(X[0])))
        # self.cov_ = np.ndarray((len(X[0]), len(X[0])))
        m = len(X)
        self.classes_, counts = np.unique(y, return_counts=True)  # classes_
        self.pi_ = counts / m
        self.cov_ = np.zeros((len(X[0]), len(X[0])))
        self.mu_ = np.ndarray((len(self.classes_), len(X[0])))
        for k, class_ in enumerate(self.classes_):
            mu = np.mean(X[y == class_], axis=0)
            self.mu_[k] = np.array(np.mean(X[y == class_], axis=0))     # mu_
            # TODO: check works np arrays - if not, add regular arrays mu, pi before
            sigma = np.transpose(X[y == class_] - self.mu_[k]) @ (X[y == class_] - self.mu_[k])
            self.cov_ += sigma  # summing all sigmas - devide after loop
        self.cov_ = self.cov_ / (m - len(self.classes_))  # cov
        self._cov_inv = np.linalg.inv(self.cov_)  # _cov_inv
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood = np.zeros((len(X), len(self.classes_)))
        for i in range(len(X)):
            cur = np.zeros((len(self.classes_)))
            for k, class_ in enumerate(self.classes_):
                a = self._cov_inv @ self.mu_[k]
                b = np.log(self.pi_[k]) - ((self.mu_[k] @ self._cov_inv @ self.mu_[k]) / 2)
                cur[k] = np.transpose(a) @ X[i] + b
            likelihood[i] = cur
        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)
