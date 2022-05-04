from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
import math
class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, self.inverses, uniques_count = np.unique(y, return_inverse=True,
                                                                return_counts=True)
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


        self.vars_ = []
        for cls in range(len(self.classes_)):
            vec = np.zeros(X.shape[1])
            for row in range(len(X)):
                if y[row] == self.classes_[cls]:
                    vec += (X[row] - self.mu_[cls]) ** 2
            vec = vec / (uniques_count[cls] - 1)
            self.vars_.append(vec)
        self.vars_ = np.array(self.vars_)
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
        # max_class = self.classes_[0]
        # max_list = []
        # for x in X:
        #     max_log = float('-inf')
        #     for cls in range(len(self.classes_)):
        #         a_k = self._cov_inv @ self.mu_[cls]
        #         b_k = math.log(self.pi_[cls] - 0.5 * (
        #                     self.mu_[cls] @ self._cov_inv @ self.mu_[cls].T))
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
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        likelihood_vals = np.zeros((X.shape[0], len(self.classes_)))
        # for row in range(len(X)):
        #     row_likelihood = []
        #     for k in range(len(self.classes_)):
        #         sqrt = 1 / ((2 * math.pi * self.vars_[k]) ** 0.5)
        #
        #         val = (-0.5 * np.linalg.norm(X[row] - self.mu_[k]) ** 2)/self.vars_[k]
        #         val = sqrt * (math.e ** val) * self.pi_[k]
        #         row_likelihood.append(val[0])
        #     likelihood_vals.append(np.array(row_likelihood))
        # return np.array(likelihood_vals)
        for k in range(len(self.classes_)):
            sqrt_val_prod = np.sqrt(np.prod(self.vars_[k]))
            sqrt_val_prod = 1 / (sqrt_val_prod * ((2 * np.pi) ** (X.shape[1] / 2)))
            for row in range(len(X)):
                center = (X[row] - self.mu_[k]) / np.sqrt(self.vars_[k])
                center = -0.5 * (np.linalg.norm(center) ** 2)
                likelihood_vals[row, k] = sqrt_val_prod * (np.e ** (center))
        return likelihood_vals



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
        return misclassification_error(self.predict(X), y)