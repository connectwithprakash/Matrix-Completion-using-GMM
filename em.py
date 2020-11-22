"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    #
    def gaussian_pdf_log(x_vector, mu_vector, var):
        d = len(mu_vector)
        return -d / 2.0 * np.log(2 * np.pi * var) - 1 / 2.0 * (
                    (x_vector - mu_vector) ** 2).sum() / var

    mu, var, weight = mixture

    X_class_weighted_density = []
    for i in range(X.shape[0]):
        mask = X[i] > 0
        class_weighted_density = []
        for j in range(mu.shape[0]):
            weighted_density = np.log(weight[j] + 1e-16) + gaussian_pdf_log(
                X[i, mask], mu[j, mask],
                var[j])
            class_weighted_density.append(weighted_density)
        X_class_weighted_density.append(class_weighted_density)

    X_class_weighted_density = np.asarray(X_class_weighted_density)
    X_probs = X_class_weighted_density - \
              logsumexp(X_class_weighted_density, axis=1)[..., None]
    X_probs = np.exp(X_probs)
    log_likelihood = logsumexp(X_class_weighted_density, axis=1).sum()

    return X_probs, log_likelihood

    # raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    d = X.shape[1]
    n_hat = post.sum(axis=0)
    p = n_hat / X.shape[0]

    mu = mixture.mu.copy()
    var = mixture.var.copy()

    for j in range(post.shape[1]):
        sse, weight = 0, 0
        for l in range(d):
            mask = X[:, l] > 0
            post_j_sum = post[mask, j].sum()
            if post_j_sum >= 1:
                mu[j, l] = np.dot(post[mask, j], X[mask, l])/post_j_sum
            sse += np.dot(((X[mask, l] - mu[j, l])**2), post[mask, j])
            weight += post_j_sum
        var[j] = sse/weight
        if var[j] < min_variance:
            var[j] = min_variance

    return GaussianMixture(mu, var, p)
    # raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_likelihood = -np.inf
    while True:
        post, new_log_likelihood = estep(X, mixture)
        mixture = mstep(X, post, mixture, 0.25)
        if (new_log_likelihood - old_log_likelihood) < 1e-6 * abs(
                new_log_likelihood):
            break
        old_log_likelihood = new_log_likelihood

    return mixture, post, new_log_likelihood
    # raise NotImplementedError

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    def gaussian_pdf_log(x_vector, mu_vector, var):
        d = len(mu_vector)
        return -d / 2.0 * np.log(2 * np.pi * var) - 1 / 2.0 * (
                    (x_vector - mu_vector) ** 2).sum() / var

    X_pred = X.copy()
    mu, var, p = mixture
    k = len(mixture.p)
    for i in range(X.shape[0]):
        mask = X[i] > 0
        post = np.asarray([np.log(p[j]) + gaussian_pdf_log(X[i, mask], mu[j, mask], var[j]) for j in range(k)])
        post = np.exp(post - logsumexp(post)[..., None])
        X_pred[i, ~mask] = np.dot(post, mu[:, ~mask])

    return X_pred
    # raise NotImplementedError
