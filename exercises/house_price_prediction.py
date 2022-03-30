import numpy

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
    prices = full_data["price"]
    bedrooms = full_data["bedrooms"]
    living_size = full_data["sqft_living"]
    full_data = full_data[bedrooms > 0]
    full_data = full_data[prices > 0]
    full_data = full_data[living_size > 0]
    fits = list(["bedrooms" ,"bathrooms", "sqft_living", "sqft_lot", "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "lat", "long", "sqft_living15", "sqft_lot15"])
    features = pd.concat([full_data[fits],
                          pd.get_dummies(full_data["floors"],
                                         drop_first=True),
                          pd.get_dummies(full_data["zipcode"],
                                         drop_first=True),
                          pd.get_dummies(full_data["date"],
                                         drop_first=True)], axis=1)
    lables = full_data["price"]
    return features, lables


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
    fig2 = px.scatter(x=X.columns, y=stats.pearsonr(X.columns[:], y),
                      title="samples and their pdf values according to the estimated gaussian",
                      labels={'x': 'Sample value',
                              'y': 'Pdf of the sample'})
    fig2.update_layout(title_x=0.5)
    fig2.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, lables = load_data(
        "/Users/sigalnaim/Desktop/IML/IML.HUJI/datasets/house_prices.csv")
    feature_evaluation(features, lables)
    #raise NotImplementedError()

    # Question 2 - Feature evaluation with respect to response
    #raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    #raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
   # raise NotImplementedError()
