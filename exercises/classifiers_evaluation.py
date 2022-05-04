import math

import numpy
import numpy as np

import IMLearn.metrics
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for title, f in [("Linearly Separable", "linearly_separable.npy"),
                     ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset

        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perc = Perceptron(True, 1000,
                          lambda P, X_tag, y_tag: (losses.append(P.loss(X, y))))
        perc.fit(X, y)
        fig1 = go.Figure()
        add_fig(fig1, losses, title)

        # Plot figure of loss as function of fitting iteration


def add_fig(fig, losses, data_title):
    fig.add_trace(go.Scatter(x=np.arange(1, len(losses) + 1, 1, dtype=int), y=losses,
                             mode="lines"))
    fig.update_layout(
        title="Losses of Perceptron fitted over " + data_title.lower() + " data as a function of itertaions until algorithm stops",
        xaxis_title="Number of iterations",
        yaxis_title="Loss (Normalized)", title_x=0.5)
    fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)
        lda = LDA()
        bayes = GaussianNaiveBayes()

        # Fit models and predict over training set
        lda.fit(X, y)
        bayes.fit(X, y)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots

        lda_prediction = lda.predict(X)
        bayes_prediction = bayes.predict(X)
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            "LDA model, accuracy: " + str(IMLearn.metrics.accuracy(y, lda_prediction)),
            "Naive Gaussian model, accuracy: " + str(
                IMLearn.metrics.accuracy(y, bayes_prediction)), ])
        fig.update_layout(title_text=f)
        #view scatters
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=dict(color=lda_prediction, symbol=y),
                       showlegend=False),
            row=1, col=1
        )
        #view ellipses and centres of graphs
        view_ellipses_and_centres(lda.mu_, lda.cov_, fig, 1)
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=dict(color=bayes_prediction, symbol=y),
                       showlegend=False),
            row=1, col=2
        )
        view_ellipses_and_centres_bayes(bayes.mu_,bayes.vars_,fig,2)

        fig.update_layout(height=600, width=1400)
        fig.show()


def view_ellipses_and_centres(mu, cov, fig, col):
    fig.add_trace(
        go.Scatter(x=mu[:, 0], y=mu[:, 1], mode="markers",
                   marker=dict(color="black", symbol='x'),
                   showlegend=False),
        row=1, col=col
    )
    for i in range(len(mu)):
        fig.add_trace(
            get_ellipse(mu[i], cov),
            row=1, col=col
        )
def view_ellipses_and_centres_bayes(mu, vars, fig, col):
    fig.add_trace(
        go.Scatter(x=mu[:, 0], y=mu[:, 1], mode="markers",
                   marker=dict(color="black", symbol='x'),
                   showlegend=False),
        row=1, col=col
    )
    for i in range(len(mu)):
        fig.add_trace(
            get_ellipse(mu[i], np.diag(vars[i])),
            row=1, col=col
        )





if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()
    X =  [[1,1], [1,2], [2,3],[2,4],[3,3],[3,4]]
    Y = [0,0,1,1,1,1]
    bayes = GaussianNaiveBayes()
    bayes.fit(np.atleast_2d(X), np.atleast_2d(Y).T)
    lda = LDA()
    lda.fit(np.atleast_2d(X), np.atleast_2d(Y).T)
    print("bye")


