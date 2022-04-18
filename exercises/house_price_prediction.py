import numpy
from sklearn.model_selection import ParameterGrid

import IMLearn.utils
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy import stats

pio.templates.default = "simple_white"


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
    full_data = pd.read_csv(filename).drop_duplicates()
    full_data = full_data[full_data["bedrooms"] > 0]
    full_data = full_data[full_data["price"] > 0]
    full_data = full_data[full_data["sqft_living"] > 0]
    full_data["floors"] = full_data["floors"].astype(float)
    full_data = full_data[full_data["floors"] > 0]
    full_data = full_data[full_data["waterfront"] >= 0]
    full_data = full_data[full_data["waterfront"] <= 1]
    full_data = full_data[full_data["bathrooms"] > 0]
    full_data = full_data[2022 >= full_data["yr_built"]]
    full_data = full_data[0 < full_data["yr_built"]]
    full_data = full_data[2022 > full_data["yr_renovated"]]
    full_data = full_data[0 <= full_data["yr_renovated"]]
    full_data = full_data[full_data["sqft_living15"] > 0]
    full_data = full_data[full_data["sqft_lot"] > 0]
    full_data = full_data[full_data["sqft_lot15"] > 0]



    dates = full_data["date"]
    splitted = dates.str.extract(r'([\d]{4})([\d]{2})([\d]{2})', expand=True)
    full_data[["buy_year", "buy_month", "buy_date"]] = splitted[[0,1,2]].fillna(0).astype(int)

    fits = list(["bedrooms" ,"buy_year", "buy_month", "buy_date", "bathrooms", "floors", "sqft_living", "sqft_lot", "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "lat", "long", "sqft_living15", "sqft_lot15"])
    full_data = full_data[full_data["buy_year"] > 0]
    features = full_data[fits]

    labels = full_data["price"]

    return features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
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
    X2, y2 = X.to_numpy(), y.to_numpy()
    corrs = []
    for i in range(len(X2[0])):
        corrs.append((__get_corr(X2[:,i], y2)))

    fig2 = px.scatter(x=X.columns, y=corrs,
                      title="features and their pearson correlation with the prices",
                      labels={'x': 'features',
                              'y': 'correlations with the price'})
    fig2.update_layout(title_x=0.5)
    fig2.show()
    #fig2.write_image(output_path)

def __get_corr(feature, prices):
    """
    a function that calculates the correlation between a current feature and the house
    prices
    """
    feature_roof, prices_roof = feature.mean(), prices.mean()
    return numpy.inner(feature - feature_roof, prices - prices_roof) /\
           ((numpy.inner(feature - feature_roof, feature - feature_roof) ** 0.5) *
            (numpy.inner(prices - prices_roof, prices - prices_roof) ** 0.5))
def __scatter_corr_feature_prices(feature, label, feature_name):
    """
    a function that displays a scatter plot showing the correlation between a feature
    and the prices
    """
    fig2 = px.scatter(x=feature, y=label,
                      title="correlation between prices and " + feature_name,
                      labels={'x': feature_name,
                              'y': "price"})
    fig2.update_layout(title_x=0.5)
    fig2.show()

def __sample_and_fit(trainX, trainY, testX, testY):
    """
    a function that trains a linear regression model over different precentages of the training data
    and displays t
    """
    sample_range = np.linspace(10, 100, 91)
    loss_mean, var_mean,gray_plus, gray_minus = [], [], [], []
    for p in sample_range:
        dev = []
        for i in range(10):
            per_train_X, per_train_Y, per_test_X, per_test_Y = IMLearn.utils.split_train_test(trainX, trainY, p / 100)
            current_linear_model = LinearRegression(False)
            current_linear_model.fit(per_train_X.to_numpy(), per_train_Y.to_numpy())

            dev.append(current_linear_model.loss(testX, testY))
        #add mean, var and confidence intervals
        dev = np.array(dev)
        loss_mean.append(dev.mean())
        var_mean.append(dev.var())
        gray_plus.append(dev.mean() + (dev.std() * 2))
        gray_minus.append(dev.mean() - (dev.std() * 2))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sample_range, y=loss_mean))
    fig2.add_trace(go.Scatter(x=sample_range, y=gray_plus, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False))
    fig2.add_trace(go.Scatter(x=sample_range, y=gray_minus, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False))
    fig2.update_layout(title="Mean Loss of price-prediction when fitting p percent of"
                             "training data", xaxis_title="percentage of training",
    yaxis_title="Mean Loss",title_x=0.5)
    fig2.show()



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, labels = load_data(
        "/Users/sigalnaim/Desktop/IML/IML.HUJI/datasets/house_prices.csv")
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(features, labels)

    # Question 3 - Split samples into training- and testing sets.
    __scatter_corr_feature_prices(features["long"], labels, "longtitude")
    __scatter_corr_feature_prices(features["sqft_living"], labels, "living room size (sqft)")
    train_X, train_Y, test_X, test_Y = IMLearn.utils.split_train_test(features, labels)



    # Question 4 - Fit model over increasing percentages of the overall training data

    __sample_and_fit(train_X, train_Y, test_X, test_Y)
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array(
        [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275,
         563867.1347574, 395102.94362135])

    print(IMLearn.metrics.mean_square_error(y_true, y_pred))


