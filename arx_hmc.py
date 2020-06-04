###############################################################################
#    Practical Bayesian System Identification using Hamiltonian Monte Carlo
#    Copyright (C) 2020  Johannes Hendriks < johannes.hendriks@newcastle.edu.a >
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

"""Estimates an ARX model using data with Gaussian noise and known model orders."""
""" allows for horseshoe prior, l1 prior, and l2 priors """

import pystan
import numpy as np
from scipy.io import loadmat
from helpers import build_input_matrix
from helpers import build_obs_matrix


def run_arx_hmc(data_path, input_order, output_order,  prior='hs'):
    """Input order gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}"""
    """Output order gives the terms a_0 * y_{k-1} + ... + a_{output_order-1}*y_{k-output_order} """
    """Priors can be 'hs', 'l1' and 'l2' """

    data = loadmat(data_path)

    y_est = data['y_estimation'].flatten()
    u_est = data['u_estimation'].flatten()
    y_val = data['y_validation'].flatten()
    u_val = data['u_validation'].flatten()


    # build regression matrix
    est_input_matrix = build_input_matrix(u_est, input_order)
    est_obs_matrix = build_obs_matrix(y_est, output_order)
    val_input_matrix = build_input_matrix(u_val, input_order)
    val_obs_matrix = build_obs_matrix(y_val, output_order)

    # trim measurement vectors to suit regression matrix
    max_delay = np.max((output_order, input_order - 1))
    y_est = y_est[int(max_delay):]
    y_val = y_val[int(max_delay):]

    # Set up parameter initialisation, initialise from +/- 40% of the maximum likelihood estimate
    def init_function():
        a_init = data['a_ML'].flatten()[1:output_order + 1]
        b_init = data['b_ML'].flatten()
        sig_e_init = data['sig_e_ML'].flatten()
        output = dict(a_coefs=a_init * np.random.uniform(0.6, 1.4, len(a_init)),
                      b_coefs=b_init * np.random.uniform(0.6, 1.4, len(b_init)),
                      sig_e=(sig_e_init * np.random.uniform(0.6, 1.4))[0],
                      shrinkage_param=np.abs(np.random.standard_cauchy(1))[0]
                      )
        return output

    # specify model file
    if prior == 'hs':
        model = pystan.StanModel(file='stan/arx.stan')
    elif prior == 'l1':
        model = pystan.StanModel(file='stan/arx_l1.stan')
    elif prior == 'l2':
        model = pystan.StanModel(file='stan/arx_l2.stan')
    else:
        print("invalid prior specified, priors can be 'hs', 'l1', or 'l2' ")

    # specify the data
    stan_data = {'input_order': int(input_order),
                 'output_order': int(output_order),
                 'no_obs_est': len(y_est),
                 'no_obs_val': len(y_val),
                 'y_est': y_est,
                 'est_obs_matrix': est_obs_matrix,
                 'est_input_matrix': est_input_matrix,
                 'val_obs_matrix': val_obs_matrix,
                 'val_input_matrix': val_input_matrix
                 }

    # perform sampling using hamiltonian monte carlo
    fit = model.sampling(data=stan_data, init=init_function, iter=6000, chains=4)

    # extract the results
    traces = fit.extract()

    return traces
