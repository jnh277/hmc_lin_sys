###############################################################################
#    Practical Bayesian System identification using Hamiltonian Monte Carlo
#    Copyright (C) 2018  Johan Dahlin < uni (at) johandahlin [dot] com >
#    Updated 2020 by Johannes Hendriks <Johannes.Hendriks@newcastle.edu.au>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
###############################################################################

""" Estimates an ARX model using Metroplis hastings or mMALA and an L2 prior on parameters """

import numpy as np
from mh_functions import get_loglike_grads
from mh_functions import build_phi_matrix
from scipy.stats import multivariate_normal
from scipy.io import loadmat

def run_arx_mh( mh2=True):
    """runs metropolis hastings for arx model, setting mh2=True uses mMALA."""

    order_a = 2
    order_b = 3
    order_ab = order_a + order_b
    order_max = np.max((order_a, order_b))
    no_params = order_ab + 1

    # Settings for MH
    no_iters = 10000
    no_burnin_iters = 1000
    step_size = 0.25
    cov_matrix = np.diag((0.057, 0.023, 0.006, 0.008, 0.001,  0.028))

    data = loadmat('data/arx_example_part_one.mat')

    y_est = data['y_estimation'].flatten()
    u_est = data['u_estimation'].flatten()

    Phi = build_phi_matrix(obs=y_est, order=(order_a, order_b), inputs=u_est)

    #¤ Metropolis-Hastings
    # Allocate variables
    theta = np.zeros((no_iters, no_params))
    grad = np.zeros((no_iters, no_params))
    hess = np.zeros((no_iters, no_params, no_params))
    loglike = np.zeros(no_iters)
    accepted = 1.0

    # Initialisation
    theta[0, :] = np.hstack((np.zeros(order_a), np.zeros(order_b), 1.0))
    loglike[0], grad[0, :], hess[0, :, :] = get_loglike_grads(y_est, Phi, theta[0, :], order_max, mh2)

    if not mh2:
        hess[0, :, :] = step_size**2 * cov_matrix

    for k in range(1, no_iters):
        # Proposal
        mean = theta[k-1, :] + 0.5 * step_size**2 * grad[k-1, :]
        cov = step_size**2 * hess[k-1, :, :]
        theta_prop = np.random.multivariate_normal(mean=mean, cov=cov)

        # Log-likelihood
        loglike_prop, grad_prop, hess_prop = get_loglike_grads(y_est, Phi, theta_prop, order_max, mh2)

        if not mh2:
            hess_prop = step_size**2 * cov_matrix

        # Accept / reject
        prop_prop = multivariate_normal.logpdf(theta_prop, mean, cov)
        mean = theta_prop + 0.5 * step_size**2 * grad_prop
        cov = step_size**2 * hess_prop
        prop_curr = multivariate_normal.logpdf(theta[k-1, :], mean, cov)

        aprob = np.exp(loglike_prop - loglike[k-1] + prop_curr - prop_prop)

        if np.random.uniform() < aprob:
            theta[k, :] = theta_prop
            loglike[k] = loglike_prop
            grad[k, :] = grad_prop
            hess[k, :, :] = hess_prop
            accepted += 1.0
        else:
            theta[k, :] = theta[k-1, :]
            loglike[k] = loglike[k-1]
            grad[k, :] = grad[k-1, :]
            hess[k, :, :] = hess[k-1, :, :]

        if np.remainder(k+1, 1000) == 0:
            print("Iteration: {} of {} done with accept rate: {}.".format(k+1, no_iters, np.round(accepted / k, 2)))

    # Save to file for plotting in R
    output = {'sigmae': theta[no_burnin_iters:no_iters, order_ab]}

    for i in range(order_a):
        a_name = 'a' + str(i+1)
        # the negative sign is needed because of how the regressor matrix was defined for the metropolis hastings function
        output.update({a_name: -theta[no_burnin_iters:no_iters, 0 + i]})

        b_name = 'b'+str(i+1)
        output.update({b_name: theta[no_burnin_iters:no_iters, order_a + i]})

    b_name = 'b0'
    output.update({b_name: theta[no_burnin_iters:no_iters, order_a + 2]})



    if mh2:
        output.update({'method':'mh2'})
    else:
        output.update({'method':'mh0'})


    # print(np.mean(theta[no_burnin_iters:no_iters, :], axis=0))

    return output

# if __name__ == "__main__":
#     run()
