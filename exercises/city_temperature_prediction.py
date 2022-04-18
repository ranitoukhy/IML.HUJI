import numpy

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

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

    data = pd.read_csv(filename, parse_dates=["Date"])
    data["Year"] = data["Year"].astype(int)
    data["Month"] = data["Month"].astype(int)
    data["Day"] = data["Day"].astype(int)
    data["Temp"] = data["Temp"].astype(float)
    data = data[data["Month"] > 0]
    data = data[data["Temp"] > -20]
    data = data[data["Month"] < 13]
    data = data[data["Day"] > 0]
    data = data[data["Day"] < 32]
    data = data[data["Year"] > 0]
    data['dayofyear'] = data['Date'].dt.dayofyear


    data = pd.concat(
        [data[[f for  f in data.columns if f not in ["Date", "City"]]],

         pd.get_dummies(data["City"])], axis=1)
    return data


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")
    israel_samples = df[df["Country"] == "Israel"]
    israel_samples["Year"] = israel_samples["Year"].astype(str)
    df["Country"] = df["Country"].astype(str)
    # Question 2 - Exploring data for specific country
    fig = px.scatter(israel_samples, x=israel_samples["dayofyear"],
                     y=israel_samples["Temp"], color="Year",
                     title="Average temperatures in israel as a function of day of year")

    fig.show()
    final_bar = israel_samples[["Month", "Temp"]]
    bars = final_bar.groupby("Month").agg("std")
    months_df = ['','January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    bars["Month"] = pd.DataFrame(months_df)
    fig2 = px.bar(bars, x='Month', y='Temp')
    fig2.update_layout(title="STD of temperatures in israel in every month")
    fig2.update_layout(xaxis_title="Month of year")
    fig2.update_layout(yaxis_title="STD of temperatures")


    fig2.show()
    # Question 3 - Exploring differences between countries
    error_mean = df.groupby(["Country", "Month"])["Temp"].agg(["mean", "std"]).reset_index()
    fig3 = px.line(error_mean, x="Month", y="mean", line_group="Country", color="Country", error_y="std")
    fig3.update_layout(title="Average temperatures of every month of each country and their STD value each month",
                       xaxis_title="Month", yaxis_title= "Average Temperature")
    fig3.show()



    # Question 4 - Fitting model for different values of `k`
    trainX, trainY, testX, testY = split_train_test(israel_samples["dayofyear"], israel_samples["Temp"])
    losses = []
    print(trainX)
    for k in range(1, 11, 1):
        model = PolynomialFitting(k)
        model.fit(trainX.to_numpy(), trainY.to_numpy())
        current_loss = np.round(model.loss(testX.to_numpy(), testY.to_numpy()), 2)
        print("k = " + str(k) + ", loss = " + str(current_loss))
        losses.append(current_loss)
    losses = np.array(losses)

    loss_bars = pd.DataFrame()
    loss_bars["k"] = pd.DataFrame([i for i in range(1,11,1)])
    loss_bars["values"] = pd.DataFrame(losses)

    fig4 = px.bar(loss_bars, x='k', y='values')
    fig4.update_layout(title="Errors of polynomial learners with different degrees")
    fig4.update_layout(xaxis_title="Polynomial degree")
    fig4.update_layout(yaxis_title="Loss")
    fig4.show()
    # Question 5 - Evaluating fitted model on different countries

    #we derived that the best polynomial degree is k = 6
    poly_model = PolynomialFitting(6)
    poly_model.fit(trainX.to_numpy(), trainY.to_numpy())
    error_countries = pd.DataFrame()
    error_countries["Country"] = pd.DataFrame([country for country in df["Country"].drop_duplicates().values if country != "Israel"])
    error_values = []
    for country in df["Country"].drop_duplicates().values:
        if country == "Israel":
            continue
        coutry_df = df[df["Country"] == country]
        error_values.append(poly_model.loss(coutry_df["dayofyear"], coutry_df["Temp"]))

    error_countries["Losses"] = pd.DataFrame(error_values)
    fig5 = px.bar(error_countries, x='Country', y='Losses', text_auto=True)
    fig5.update_layout(title="Error of temperature prediction of each country according to model learned in israel")
    fig5.update_layout(xaxis_title="Country")
    fig5.update_layout(yaxis_title="Loss Error")
    fig5.show()



