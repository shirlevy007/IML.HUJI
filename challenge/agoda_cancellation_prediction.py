from sklearn.base import BaseEstimator

from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd


def cancellacion(c_policy, days_before_cancel, price_per_night, price_total, no_show=False):
    if (c_policy == "UNKNOWN"):
        return 0
    if no_show:
        days_before_cancel = 0
    i = 0
    days = 0
    payment = 0
    last_payment = 0
    ns = 0
    while (c_policy[i:]):
        temp = 0
        while (c_policy[i].isdigit()):
            temp = temp * 10 + int(c_policy[i])
            i += 1
        if (c_policy[i] == "D"):
            ns += 1
            days = temp
            i += 1
            if (days_before_cancel > days):
                return payment
        elif (c_policy[i] == "N"):
            ns -= 1
            last_payment = payment
            payment = temp * price_per_night
            i += 1
        elif (c_policy[i] == "P"):
            ns -= 1
            last_payment = payment
            payment = (temp / 100) * price_total
            i += 1
        if (len(c_policy)>i):
            if (c_policy[i] == "_"):
                i += 1  # the tap "_"
        else:
            break
    if no_show:
        return payment
    if ns:
        return last_payment
    return payment


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    features = full_data[["h_booking_id",
                          "hotel_id",
                          "accommadation_type_name",
                          "hotel_star_rating",
                          "customer_nationality"]]
    labels = full_data["cancellation_datetime"]

    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    # np.random.seed(0)
    #
    # # Load data
    # df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    # train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)
    #
    # # Fit model over data
    # estimator = AgodaCancellationEstimator().fit(train_X, train_y)
    #
    # # Store model predictions over test set
    # evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")

    print(cancellacion("130D1N_45D90P_100P", 42, 500, 2000))
