from pygments.lexers import go

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    dataframe = pd.read_csv(filename, parse_dates=['Date'])
    dataframe = dataframe.dropna()
    dataframe = dataframe.drop_duplicates()

    dataframe['DayOfYear'] = dataframe['Date'].dt.dayofyear
    dataframe = dataframe[dataframe['Temp'] > -70]

    return dataframe


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("..\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_israel = df[df['Country'] == "Israel"]
    year_colors = df_israel["Year"].astype(str)
    fig2_1 = px.scatter(data_frame=df_israel,
                        x='DayOfYear', y='Temp',
                        color=year_colors,
                        title="Israel's average daily temperature change as a function of the day of year",
                        labels={"DayOfYear": "Day of year", "Temp": "Temperature"})
    fig2_1.show()
    # print('done 2_1')

    bymonths_std = df_israel.groupby(['Month']).agg('std')

    fig2_2 = px.bar(bymonths_std, x=bymonths_std.index, y="Temp",
                    title="Israel's standard deviation of the daily temperatures by months",
                    labels={"y": "Temperature's Standard Deviation"})

    fig2_2.show()
    # print('done 2_2')

    # Question 3 - Exploring differences between countries

    country_month_mean = df.groupby(['Country', 'Month']).Temp.agg(['mean', 'std']).reset_index()
    fig3 = px.line(country_month_mean,
                   x=country_month_mean['Month'], y=country_month_mean['mean'],
                   error_y=country_month_mean['std'],
                   color=country_month_mean['Country'],
                   title="Average monthly temperature, color coded by the country",
                   labels={"mean": "Temperature's mean"})
    fig3.show()
    # print('done 3')


    # Question 4 - Fitting model for different values of `k`
    israel_X = df_israel.drop(['Temp'], axis=1)
    israel_y = pd.Series(df_israel['Temp'])
    train_X, train_y, test_X, test_y = split_train_test(israel_X, israel_y)

    loss = []
    for k in range(1, 11):
        modelP = PolynomialFitting(k)

        fit_k = modelP.fit(train_X['DayOfYear'].to_numpy(), train_y.to_numpy())
        loss_k = (fit_k.loss(test_X['DayOfYear'].to_numpy(), test_y.to_numpy()))
        round_loss_k = np.round(loss_k, 2)
        loss.append(round_loss_k)

        print(round_loss_k)

    fig4 = px.bar(x=range(1, 11), y=loss,
                  title="Test error (of model) recorded for each value of k",
                  labels={"x": "Degree", "loss": "Loss"})
    fig4.show()
    # print('done 4')


    # Question 5 - Evaluating fitted model on different countries

    chosen_k = 5  # minimal loss
    modelP = PolynomialFitting(chosen_k)
    modelP = modelP.fit(df_israel['DayOfYear'].to_numpy(), df_israel['Temp'].to_numpy())

    countries = ['Israel']
    modelErr = []
    for country in df['Country']:
        if country in countries:
            continue
        countries.append(country)
        # print(country)
        df_country = df[df['Country'] == country]
        country_loss = modelP.loss(df_country['DayOfYear'], df_country['Temp'])
        modelErr.append(country_loss)
    countries.remove('Israel')


    fig5 = px.bar(x=countries, y=modelErr,
                  title="model’s error over each of the countries (except IL) for k=5",
                  labels={"x": "Country", "y": "Loss"})

    fig5.show()
    # print('done 5')
