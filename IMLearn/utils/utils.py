from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    # perm = np.random.permutation(len(X))
    # train_X, train_Y = X.iloc[perm[:int(train_proportion * len(X))]],  y.iloc[perm[:int(train_proportion * len(X))]]
    # test_X, test_Y = X.iloc[perm[int(train_proportion * len(X)):]],  y.iloc[perm[int(train_proportion * len(X)):]]
    # return train_X, train_Y, test_X, test_Y
    num_samples = len(X)
    random_indices = np.random.permutation(num_samples)
    train_size = np.ceil(train_proportion * num_samples).astype(int)
    test_size = num_samples - train_size

    test_indices = random_indices[:test_size]
    train_indices = random_indices[test_size:]

    return X.iloc[train_indices], y.iloc[train_indices], X.iloc[test_indices], y.iloc[
        test_indices]





def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
