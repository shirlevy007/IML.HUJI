import numpy as np
# from ...base import BaseEstimator #used to be ... and changed
from typing import Callable, NoReturn
from IMLearn.base import BaseEstimator
import IMLearn.learners.classifiers.decision_stump
from IMLearn.metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.weights_ = np.zeros(self.iterations_)
        m = len(y)
        # self.models_ = [None]*self.iterations_
        self.D_ = np.ones(m) / m
        models = []
        for t in range(self.iterations_):
            h_t = self.wl_()
            h_t.fit(X, self.D_*y) # = H_t
            y_pred = h_t.predict(X) # = H_t (Xi)
            epsilon = np.sum((np.abs(y_pred - y)/2) * self.D_) # = Epsilon_t
            self.weights_[t] = np.log((1/epsilon)-1) / 2 # = w_i
            self.D_ *= np.exp((-1) * self.weights_[t] * y * y_pred) # =sum Di(t+1) = D(t+1)
            self.D_ /= np.sum(self.D_)
            models.append(h_t)
        self.models_ = models


    def _predict(self, X):
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
        return self.partial_predict(X, self.iterations_)

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
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y_pred = np.zeros(len(X))
        for t in range(T):
            y_pred += (self.models_[t].predict(X)) * self.weights_[t]
        return np.sign(y_pred)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.partial_predict(X, T)
        # return misclassification_error(y, y_pred)
        return sum(abs(y - y_pred)/2) / len(y)