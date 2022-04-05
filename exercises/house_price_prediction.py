from plotly.io import orca

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

POSSIBLE_ZEROS = ['waterfront', 'view', 'sqft_basement', 'zipcode']


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    dataframe = pd.read_csv(filename)
    dataframe = dataframe.dropna()
    # dataframe = dataframe.drop_duplicates()
    # dataframe = dataframe.filna(0, inplace=True)


    dataframe = preprosses(dataframe)

    dataframe = dataframe[dataframe['price'] > 0]

    prices_real = pd.Series(dataframe['price'])
    # prices_real = prices_real.drop(prices_real[prices_real['prices'] == 0], inplace=True)
    dataframe = dataframe.drop(['price'], axis=1)

    return dataframe, prices_real


def year_max(row):
    yr_built = int(row["yr_built"])
    yr_renovated = int(row["yr_renovated"])
    row["yr_max"] = max(yr_renovated, yr_built)
    return row


def preprosses(df):
    # df = df.apply(dates_format, axis="columns")
    df["yr_max"] = df[["yr_renovated", "yr_built"]].max(axis=1)
    df = df.drop(columns=['id', 'long', 'lat', 'date', 'yr_built', 'yr_renovated'])
    df = pd.get_dummies(data=df, columns=['zipcode'])
    # df = df.loc( [:, df.columns != ['id', 'long', 'lat', 'date', 'yr_built', 'yr_renovated']])
    for column in df.columns:
        if 'zipcode' in column:
            continue
        if 'price' in column:
            continue
        limit = 0 if column in POSSIBLE_ZEROS else 1
        df.drop(df[df[column] < limit].index, inplace=True)
    return df
    # df = pd.get_dummies(df, columns=['zipcode'])
    # df.drop('zipcode', axis=1, inplace=True)




# def dates_format(row):
#     date_time = pd.to_datetime(row["date"])
#     row["date"] = date_time.timestamp()
#     return row


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # pc_temp = []
    # outcome = y.to_numpy()
    # outcome = y["price"].to_numpy()
    y_std = np.std(y)
    # print(y_std)
    for col in X.columns:
        # col = X[i].to_numpy()
        if "zipcode" in col:
            continue
        cov = np.cov(X[col], y)[0][1]
        X_std = np.std(X[col])
        pc = cov / (X_std * y_std)
        # pc_temp.append(pc)
        # fig = go.Figure([go.Scatter(x=col, y=y, mode="markers", name="corr fitures prices", marker=dict(color="black"),
        #                           showlegend=False)], rows=1, cols=i + 1)
        fig = go.Figure([go.Scatter(x=X[col], y=y,
                                    name=r"Correlation feature {0} "
                                         r"and Prices is {1}".format(col, pc),
                                    mode='markers',
                                    marker=dict(color="LightSkyBlue"),
                                    showlegend=False)], layout=dict(
            title=r"Correlation Between feature {0} "
                  r"and Prices is {1}".format(col, pc)))
        # fig.show()
        # fig.update_xaxes(title=col)
        # fig.update_yaxes(title="price")
        # pio.write_image(fig=fig, file=output_path + r"\{}.png".format(col), format='png')
        fig.write_image(output_path+"\\"+col+".png")

    print("done 2")


if __name__ == '__main__':
    np.random.seed(0)

    # Store model predictions over test set

    # load_data(..IML.HUJI/datasets/house_prices.csv)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, prices = load_data("..\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(df, prices, "../exercises")
    feature_evaluation(df, prices, r"C:\Users\shirl\Documents\plots")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df, prices)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    loss_std = []
    loss_mean = []
    for p in range(10, 101):
        loss = []
        for i in range(10):
            train_samp = train_X.sample(frac=p / 100)
            train_resp = train_y[train_samp.index]

            fit_i = LinearRegression().fit(train_samp.to_numpy(), train_resp)
            loss_i = fit_i.loss(test_X.to_numpy(), test_y.to_numpy())
            # np.append(loss, loss_i)
            loss.append(loss_i)
        loss_mean.append(np.array(loss).mean())
        # np.append(loss_mean, np.mean(loss))
        loss_std.append(np.array(loss).std())
        # np.append(loss_std, np.std(loss))
    loss_mean = np.array(loss_mean)
    loss_std = np.array(loss_std)

    # print("Loss mean: " + str(loss_mean))

    go.Figure([go.Scatter(x=list(range(10, 101)), y=loss_mean, mode='markers+lines', showlegend=False),
               go.Scatter(x=list(range(10, 101)), y=loss_mean - 2 * loss_std, fill=None, mode="lines",
                          line=dict(color="lightgrey"), showlegend=False),
               go.Scatter(x=list(range(10, 101)), y=loss_mean + 2 * loss_std, fill='tonexty', mode="lines",
                          line=dict(color="lightgrey"), showlegend=False)],
              layout=go.Layout(
                  title=r"$\text{average loss as function of training size }$",
                  xaxis_title="$\\text{percent taken from train data}$",
                  yaxis_title="r$\\text{mean loss}$",
                  height=500)).show()



    # loss_std = np.array([])
    # loss_mean = np.array([])
    # for p in range(10, 101):
    #     loss = np.array([])
    #     for i in range(10):
    #         train_X = train_samples.sample(frac=p / 100)
    #         train_y = train_response[train_response.index.isin(train_X.index)]
    #         fit_i = LinearRegression().fit(train_X.to_numpy(), train_y)
    #         loss_i = fit_i.loss(test_samples.to_numpy(), test_response.to_numpy())
    #         np.append(loss, loss_i)
    #     np.append(loss_mean, np.mean(loss))
    #     np.append(loss_std, np.std(loss))
    #
    # loss = np.array(loss)
    # loss_mean = loss.mean()
    # print("Loss mean: " + str(loss_mean))
