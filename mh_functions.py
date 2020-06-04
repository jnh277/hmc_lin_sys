import os
import numpy as np
from scipy.stats import norm

def build_phi_matrix(obs, order, inputs=False):
    """Builds the regressor matrix for least squares."""
    no_obs = len(obs)
    if isinstance(inputs, bool):
        phi = np.zeros((no_obs, order + 1))
        for i in range(order, no_obs):
            phi[i, :] = obs[range(i, i - order - 1, -1)]
        return phi[order:, :]
    else:
        phi = np.zeros((no_obs, order[0] + order[1]))
        for i in range(int(np.max(order)), no_obs):
            phi[i, :] = np.hstack((obs[range(i-1, i - order[0] - 1, -1)],
                                   inputs[range(i-1, i - order[1] - 1, -1)]))
        return phi[int(np.max(order)):, :]

def get_loglike_grads(y, Phi, theta, order_max, mh2=True):
    n = len(theta) - 1
    grad = np.zeros(len(theta))
    hess = np.zeros((len(theta), len(theta)))

    ll = np.sum(norm.logpdf(y[order_max:], theta[0:n] @ Phi.T, theta[n]))
    ll += compute_log_prior(theta)

    if mh2:
        quad_form = np.linalg.norm(y[order_max:] - theta[0:n] @ Phi.T)

        grad[0:n] = theta[n]**(-2) * (y[order_max:] - theta[0:n] @ Phi.T) @ Phi
        # for i in range(n):
        #     grad[i] = theta[n]**(-2) * np.sum((y[order_max:] - theta[0:n] @ Phi.T) @ Phi[:, i])
        grad[n] = theta[n]**(-3) * quad_form - theta[n]**(-1)
        grad += compute_log_prior_grad(theta)

        hess[0:n, 0:n] = -theta[n]**(-2) * Phi.T @ Phi
        hess[n, n] = theta[n]**(-2) - 3.0 * theta[n]**(-4) * quad_form
        hess += compute_log_prior_hessian(theta)
        hess = -np.linalg.inv(hess)

        # Correct Hessian
        evd = np.linalg.eig(hess)
        ev_matrix = np.abs(evd[0])
        for i, value in enumerate(ev_matrix):
            ev_matrix[i] = value
        hess = evd[1] @ np.diag(ev_matrix) @ np.linalg.inv(evd[1])

    return ll, hess @ grad, hess



def compute_log_prior(theta):
    n = len(theta)
    log_prior = 0.0
    for i in range(n-1):
        log_prior += norm_logpdf(theta[i], 0.0, 1.0)
    log_prior += cauchy_logpdf(theta[n-1], 0.0, 1.0)
    return log_prior

def compute_log_prior_grad(theta):
    n = len(theta)
    gradient = np.zeros(n)
    for i in range(n-1):
        gradient[i] = norm_logpdf_gradient(theta[i], 0.0, 1.0)
    gradient[n-1] = cauchy_logpdf_gradient(theta[n-1], 0.0, 1.0)
    return gradient

def compute_log_prior_hessian(theta):
    n = len(theta)
    hessian = np.zeros((n, n))
    for i in range(n-1):
        for j in range(n-1):
            if i == j:
                hessian[i, j] = norm_logpdf_hessian(theta[i], 0.0, 1.0)
            else:
                hessian[i, j] = norm_logpdf_gradient(theta[i], 0.0, 1.0)
                hessian[i, j] *= norm_logpdf_gradient(theta[j], 0.0, 1.0)

    i = n - 1
    hessian[i, i] = cauchy_logpdf_hessian(theta[n-1], 0.0, 1.0)
    for j in range(n-1):
        hessian[i, j] = cauchy_logpdf_gradient(theta[i], 0.0, 1.0)
        hessian[j, i] = cauchy_logpdf_gradient(theta[i], 0.0, 1.0)
        hessian[i, j] = norm_logpdf_gradient(theta[j], 0.0, 1.0)
        hessian[i, j] = norm_logpdf_gradient(theta[j], 0.0, 1.0)

    return hessian

def cauchy_logpdf(param, loc, scale):
    """ Computes the log-pdf of the Cauchy distribution.

        Args:
            param: value to evaluate in
            loc: loc
            scale: scale

        Returns:
            A scalar with the value of the log-pdf.

    """
    term1 = np.pi * scale
    term2 = 1.0 + (param - loc)**2 * scale**(-2)
    return -np.log(term1) - np.log(term2)

def cauchy_logpdf_gradient(param, loc, scale):
    """ Computes the gradient of the log-pdf of the Cauchy distribution.

        Args:
            param: value to evaluate in
            loc: loc
            scale: scale

        Returns:
            A scalar with the value of the gradient of the log-pdf.

    """
    term2 = 1.0 + (param - loc)**2 * scale**(-2)
    return -scale**(-2) * (param - loc) / term2

def cauchy_logpdf_hessian(param, loc, scale):
    """ Computes the Hessian of the log-pdf of the Cauchy distribution.

        Args:
            param: value to evaluate in
            loc: loc
            scale: scale

        Returns:
            A scalar with the value of the Hessian of the log-pdf.

    """
    term2 = 1.0 + (param - loc)**2 * scale**(-2)
    return -scale**(-2) / term2 + scale**(-4) * (param - loc)**2 / term2**2

def norm_logpdf(parm, mean, stdev):
    """ Computes the log-pdf of the Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean
            stdev: standard deviation

        Returns:
            A scalar with the value of the log-pdf.

    """
    quad_term = -0.5 / (stdev**2) * (parm - mean)**2
    return -0.5 * np.log(2 * np.pi * stdev**2) + quad_term

def norm_logpdf_gradient(parm, mean, stdev):
    """ Computes the gradient of the log-pdf of the Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean
            stdev: standard deviation

        Returns:
            A scalar with the value of the gradient of the log-pdf.

    """
    return -(mean - parm) / stdev**2

def norm_logpdf_hessian(parm, mean, stdev):
    """ Computes the Hessian of the log-pdf of the Gaussian distribution.

        Args:
            parm: value to evaluate in
            mean: mean
            stdev: standard deviation

        Returns:
            A scalar with the value of the Hessian of the log-pdf.

    """
    return -1.0 / stdev**2