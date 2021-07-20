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

import stan
import numpy as np
from scipy.io import loadmat
from helpers import build_input_matrix
from helpers import build_obs_matrix
import pickle
from pathlib import Path


def run_arx_hmc(data_path, input_order, output_order, prior='hs', hot_start=False, iter=2000):
    """Input order gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}"""
    """Output order gives the terms a_0 * y_{k-1} + ... + a_{output_order-1}*y_{k-output_order} """
    """Priors can be 'hs', 'l1' and 'l2', and in a hacky way it can also be 'st' which represents using a student t measurement noise """
    """hot_start == True will start within 40% of the maximum likelihood results"""

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
    if hot_start == True:
        if prior == 'st':
            def init_function():
                a_ML = data['a_ML_reg'].flatten()[1:output_order + 1]
                b_ML = data['b_ML_reg'].flatten()
                sig_e_ML = data['sig_e_ML_reg'].flatten()
                output = dict(sig_e=(sig_e_ML * np.random.uniform(0.8, 1.2))[0],
                              a_coefs=a_ML * np.random.uniform(0.8, 1.2, len(a_ML)),
                              b_coefs=b_ML * np.random.uniform(0.8, 1.2, len(b_ML)),
                              shrinkage_param=np.abs(np.random.standard_cauchy(1))[0]
                              )
                return output
        else:
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
    else:
        def init_function():
            output = dict(a_coefs=np.zeros((output_order)),
                          b_coefs=np.zeros((input_order)),
                          shrinkage_param=np.abs(np.random.standard_cauchy(1))[0]
                          )
            return output

    # specify model file
    if prior == 'hs':
        f = open('stan/arx.stan', 'r')
        model_code = f.read()
    elif prior == 'l1':
        f = open('stan/arx_l1.stan', 'r')
        model_code = f.read()
    elif prior == 'l2':
        f = open('stan/arx_l2.stan', 'r')
        model_code = f.read()
    elif prior == 'st':
        f = open('stan/arx_st.stan', 'r')
        model_code = f.read()
    else:
        print("invalid prior specified, priors can be 'hs', 'l1', 'l2' or 'st ")

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
    posterior = stan.build(model_code, data=stan_data)
    init = [init_function(),init_function(),init_function(),init_function()]
    traces = posterior.sample(init=init, num_samples=iter, num_warmup=2000, num_chains=4)

    return traces
