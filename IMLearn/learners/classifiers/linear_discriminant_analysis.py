import math
from typing import NoReturn

import pandas

from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, self.inverses, uniques_count = np.unique(y, return_inverse=True,return_counts=True)
        self.pi_ = uniques_count / len(y)
        self.mu_ = []
        for cls in range(len(self.classes_)):
            vec = np.zeros(X.shape[1])
            for row in range(len(X)):
                if y[row] == self.classes_[cls]:
                    vec += X[row]
            vec = vec / uniques_count[cls]
            self.mu_.append(vec)
        self.mu_ = np.array(self.mu_)
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        for row in range(len(X)):
            centered = np.atleast_2d(X[row] - self.mu_[self.inverses[row]])
            self.cov_ += centered.T @ centered
        self.cov_ = np.array(self.cov_) / (len(X) - len(self.classes_))

        self._cov_inv = np.linalg.inv(self.cov_)
        self.fitted_ = True







    def _predict(self, X: np.ndarray) -> np.ndarray:
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
        # max_list = []
        # for x in X:
        #     max_log = float('-inf')
        #     max_class = self.classes_[0]
        #     for cls in range(len(self.classes_)):
        #         a_k = self._cov_inv @ self.mu_[cls]
        #         b_k = math.log(self.pi_[cls]-0.5 * (self.mu_[cls] @ self._cov_inv @ self.mu_[cls].T))
        #         if a_k @ x + b_k > max_log:
        #             max_log = a_k @ x + b_k
        #             max_class = self.classes_[cls]
        #     max_list.append(max_class)
        # return np.ndarray(max_list)
        likelis = self.likelihood(X)
        pred = []
        for x in likelis:
            pred.append(self.classes_[np.argmax(x)])
        return np.array(pred)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        sqrt = ((2 * math.pi) ** len(self.classes_)) * np.linalg.det(self.cov_)
        sqrt = 1 / sqrt
        likelihood_vals = []
        for row in range(len(X)):
            row_likelihood = []
            for k in range(len(self.classes_)):
                val = -0.5 * (X[row] - self.mu_[k]) @ self._cov_inv @ (X[row] - self.mu_[k]).T
                val = sqrt * (math.e ** val) * self.pi_[k]
                row_likelihood.append(val)
            likelihood_vals.append(np.array(row_likelihood))
        return np.array(likelihood_vals)



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
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
