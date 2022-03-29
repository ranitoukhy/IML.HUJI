from __future__ import annotations

import math

import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        expectency = variance = 0
        for sample in X:
            expectency += sample
        if not X.size == 0:
            expectency = expectency / X.size
        for sample in X:
            variance += (sample - expectency) ** 2
        if self.biased_ and X.size:
            variance = variance / X.size
        elif not self.biased_ and X.size > 1:
            variance = variance / (X.size - 1)
        self.mu_, self.var_ = expectency, variance
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        pdfed_samples = []
        for val in X:
            pdfed_samples.append(self.__cal_density(val))
        return np.array(pdfed_samples)

    def __cal_density(self, val):
        # a function that calculates the pdf of each value
        if self.var_ == 0:
            return 0
        return (math.e ** -(((val - self.mu_) ** 2)/(2 * self.var_)))/(1/math.sqrt(2 * math.pi * self.var_))

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # calculate the log-likelihood according to the formula found in the
        # course book
        sample_sum = 0
        for val in X:
            sample_sum += (val - mu) ** 2
        sample_sum = sample_sum / -(2 * sigma )
        divide_by = math.pow(2 * math.pi * sigma, X.size / 2)
        return sample_sum - math.log(divide_by)



class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        centered_X_matrix = X - np.tile(self.mu_, (X.shape[0], 1))
        divide_by = X.shape[0] - 1
        self.cov_ = (centered_X_matrix.T @ centered_X_matrix) / (divide_by)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        # calculate the pdf according to the formula found in the course book
        pdf_values = []
        for x in X:
            mult = -0.5 * (x - self.mu_).T @ np.linalg.inv(self.cov_) @ (x - self.mu_)
            divide_by = math.sqrt(((2 * math.pi) ** x.shape[0]) * np.linalg.det(self.cov_))
            pdf_values.append((math.e ** mult) / divide_by)
        return np.array(pdf_values)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        #calculate the log-likelihood according to the formula solved in the
        #theoritcal part
        cov_det = np.linalg.det(cov)
        cov_inverse = np.linalg.inv(cov)
        inner_mult = 0
        for vec in X:
            inner_mult += -0.5 * (vec - mu).T @ cov_inverse @ (vec - mu)
        return inner_mult - (X.size / 2) * math.log(cov_det) - X.size * X.shape[1] * 0.5 * math.log(2 * math.pi)


