from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def f_func(x, eps=0):
    res = (x + 3)*(x + 2)*(x + 1)*(x - 1)*(x - 2) + eps
    return res



def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    X = np.linspace(-1.2, 2, n_samples)
    eps = np.random.normal(0, noise, n_samples)
    y_pred = f_func(X)
    y_pred_noised = f_func(X, eps)

    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y_pred_noised), 2 / 3)
    X_train, y_train, X_test, y_test = X_train.iloc[:, 0].to_numpy(), y_train.to_numpy(), X_test.iloc[:, 0].to_numpy(), y_test.to_numpy()

    fig1 = go.Figure(
        [go.Scatter(x=X, y=y_pred, mode="markers", marker=dict(color="violet"),
                    name=r'True (noiseless) model'),
         go.Scatter(x=X_train, y=y_train, mode="markers", marker=dict(color="orange"),
                    name=r'Train model with Noise'),
         go.Scatter(x=X_test, y=y_test, mode="markers", marker=dict(color="green"),
                    name=r'Test model with Noise')
         ])
    fig1.update_layout(title=f"True (noiseless) model, train&test sets of f. Samples = {n_samples}, Noise = {noise}",
                        xaxis_title="x",
                        yaxis_title="f(x)")
    fig1.show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    train_scores, val_scores = [], []
    for k in range(11):
        poly_fit = PolynomialFitting(k)

        train_score, val_score = cross_validate(poly_fit, X_train, y_train, mean_square_error)
        train_scores.append(train_score)
        val_scores.append(val_score)

    fig2 = go.Figure(
        [go.Scatter(x=list(range(11)), y=train_scores, mode='lines+markers', marker=dict(color="violet"),
                    name=r'Train err'),
         go.Scatter(x=list(range(11)), y=val_scores, mode='lines+markers', marker=dict(color="orange"),
                    name=r'Validation err'),
         ])
    fig2.update_layout(title=f"Average train&validation scores as function of polynomial degree k in 5-fold cross-validation",
                       xaxis_title="k degree",
                       yaxis_title="Error average")
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error

    k_opt = np.argmin(val_scores) #lowest validation error was achieved
    poly = PolynomialFitting(int(k_opt))
    poly.fit(X_train, y_train)

    # test_score = poli_fit.loss(X_test, y_test)
    test_score = poly.loss(X_test, y_test)
    test_score = round(test_score, 2)
    # test_score = np.round(poli_fit.loss(X_test, y_test))

    print(f"Samples = {n_samples}, noise =  {noise}")
    print(f"optimal k is {k_opt} with test error of {test_score}")








def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions

    X,y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test = X[:n_samples, :], X[n_samples:, :]
    y_train, y_test = y[:n_samples], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    prange = np.linspace(0.01, 3, num=n_evaluations)   # possible range
    ridge_train_scores, ridge_val_scores = [], []
    lasso_train_scores, lasso_val_scores = [], []
    for lam in prange:
        ridge = RidgeRegression(lam)
        lasso = Lasso(alpha=lam)
        r_train, r_val = cross_validate(ridge, X_train, y_train, mean_square_error)
        l_train, l_val = cross_validate(lasso, X_train, y_train, mean_square_error)
        ridge_train_scores.append(r_train)
        ridge_val_scores.append(r_val)
        lasso_train_scores.append(l_train)
        lasso_val_scores.append(l_val)

    fig7 = go.Figure(
        [go.Scatter(x=prange, y=ridge_train_scores, mode='lines', marker=dict(color="violet"),
                    name=r'Ridge train err'),
         go.Scatter(x=prange, y=ridge_val_scores, mode='lines', marker=dict(color="blue"),
                    name=r'Ridge validation err'),
         go.Scatter(x=prange, y=lasso_train_scores, mode='lines', marker=dict(color="green"),
                    name=r'Lasso train err'),
         go.Scatter(x=prange, y=lasso_val_scores, mode='lines', marker=dict(color="orange"),
                    name=r'Lasso validation err'),
         ])
    fig7.update_layout(title=f"Ridge train&validation errors as function of the tested regularization parameter value",
                       xaxis_title="Lambda := tested regularization parameter value",
                       yaxis_title="Error")
    fig7.show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    ridge_min = np.argmin(ridge_val_scores)
    lasso_min = np.argmin(lasso_val_scores)
    ridge_best_lam = prange[ridge_min]
    lasso_best_lam = prange[lasso_min]
    print(f"Best regularization parameter for the Ridge:  {ridge_best_lam},\n"
          f"Best regularization parameter for the Lasso:  {lasso_best_lam}")

    r_model = RidgeRegression(ridge_best_lam)
    r_model.fit(X_train, y_train)
    r_best = r_model.predict(X_test)
    ridge_err = mean_square_error(y_test, r_best)
    print(f"Ridge test error is {ridge_err}")

    l_model = Lasso(lasso_best_lam)
    l_model.fit(X_train, y_train)
    l_best = l_model.predict(X_test)
    lasso_err = mean_square_error(y_test, l_best)
    print(f"Lasso test error is {lasso_err}")

    least_squares = LinearRegression()
    least_squares.fit(X_train, y_train)
    ls_best = least_squares.predict(X_test)
    ls_err = mean_square_error(y_test, ls_best)
    print(f"Least Squares test error is {ls_err}")



if __name__ == '__main__':
    np.random.seed(0)

    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)

    select_regularization_parameter()
