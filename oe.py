###############################################################################
#    Practical Bayesian Linear System Identification using Hamiltonian Monte Carlo
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

"""Estimates an OE model using data with Gaussian noise and horeshoe priors."""

import pystan
import numpy as np
from scipy.io import loadmat
import pickle
from pathlib import Path


# specific data path
data_path = 'data/example3_oe.mat'
input_order = 4
output_order = 3

def run_oe_hmc(data_path, input_order, output_order, hot_start=False, iter=6000, OL=False):
    """ Input order gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}"""
    """ Output order gives the terms a_0 * y_{k-1} + ... + a_{output_order-1}*y_{k-output_order} """
    """ hot_start=True will initialise at maximum likelihood results"""
    data = loadmat(data_path)

    y_est = data['y_estimation'].flatten()
    u_est = data['u_estimation'].flatten()

    ## TODO: FIX this hack
    # y_val = data['y_validation'].flatten()
    # u_val = data['u_validation'].flatten()
    y_val = data['y_estimation'].flatten()
    u_val = data['u_estimation'].flatten()

    # Run Stan
    if hot_start:
        def init_function():
            f_true = data['f_ml'].flatten()[1:output_order+1]
            b_true = data['b_ml'].flatten()
            sig_e = data['sig_e'].flatten()
            output = dict(f_coefs=np.flip(f_true),# * np.random.uniform(0.8, 1.2, len(f_true)),
                          b_coefs=np.flip(b_true),#* np.random.uniform(0.8, 1.2, len(b_true)),
                          r=(sig_e)[0],# * np.random.uniform(0.8, 1.2))[0],
                          )
            return output
    else:
        def init_function():    ## TODO: uncomment this
            # sig_e = data['sig_e'].flatten()
            # output = dict(r=(sig_e * np.random.uniform(0.8, 1.2))[0],
            #               )
            output = dict()
            return output

    if OL:
        model_path = 'stan/oe_OL.pkl'
        if Path(model_path).is_file():
            model = pickle.load(open(model_path, 'rb'))
        else:
            model = pystan.StanModel(file='stan/oe_OL.stan')
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
    else:
        model_path = 'stan/oe.pkl'
        if Path(model_path).is_file():
            model = pickle.load(open(model_path, 'rb'))
        else:
            model = pystan.StanModel(file='stan/oe.stan')
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)

    stan_data = {'input_order': int(input_order),
                 'output_order': int(output_order),
                 'no_obs_est': len(y_est),
                 'y_est': y_est,
                 'y_val': y_val,
                 'u_est':u_est,
                 'u_val':u_val,
                 'no_obs_val': len(y_val),
                 }

    fit = model.sampling(data=stan_data, init=init_function, iter=iter, chains=4)
    traces = fit.extract()

    return (fit, traces)
