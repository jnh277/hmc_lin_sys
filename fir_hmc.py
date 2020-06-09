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

"""Estimates an FIR model using data with Gaussian noise."""
import pystan
import numpy as np
from scipy.io import loadmat
from helpers import build_input_matrix
from pathlib import Path
import pickle




def run_fir_hmc(data_path, input_order,  prior='tc', hot_start=False):
    """ Input order gives the terms # gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}"""
    """ prior can be 'tc' for tuned correlated kernel or 'hs' for the horseshoe sparesness prior """
    """ hot start will use least squares values as starting point """
    data = loadmat(data_path)

    y_est = data['y_estimation'].flatten()
    u_est = data['u_estimation'].flatten()
    y_val = data['y_validation'].flatten()
    u_val = data['u_validation'].flatten()


    # build regression matrix
    est_input_matrix = build_input_matrix(u_est, input_order)
    val_input_matrix = build_input_matrix(u_val, input_order)

    # trim measurement vectors to suit regression matrix
    max_delay = (input_order-1)
    y_est = y_est[int(max_delay):]
    y_val = y_val[int(max_delay):]

    # calcualte an intial guess using least squares (ML)
    if hot_start:
        Ainv = np.linalg.pinv(est_input_matrix)
        b_init = np.matmul(Ainv, y_est)
    else:
        b_init = np.zeros((input_order))

    # Run Stan
    def init_function():
        sig_e = data['sig_e'].flatten()
        output = dict(b_coefs=b_init * np.random.uniform(0.8, 1.2, len(b_init)),
                      sig_e=(sig_e * np.random.uniform(0.8, 1.2))[0],
                      b_coefs_hyperprior=np.abs(np.random.standard_cauchy(len(b_init))),
                      shrinkage_param=np.abs(np.random.standard_cauchy(1))[0]
                      )
        return output

    stan_data = {'input_order': int(input_order),
                 'no_obs_est': len(y_est),
                 'no_obs_val': len(y_val),
                 'y_est': y_est,
                 'est_input_matrix': est_input_matrix,
                 'val_input_matrix': val_input_matrix
                 }

    # specify model file
    if prior == 'hs':
        model_path = 'stan/fir_hs.pkl'
        if Path(model_path).is_file():
            model = pickle.load(open(model_path, 'rb'))
        else:
            model = pystan.StanModel(file='stan/fir.stan')
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
    elif prior == 'tc':
        model_path = 'stan/fir_tc.pkl'
        if Path(model_path).is_file():
            model = pickle.load(open(model_path, 'rb'))
        else:
            model = pystan.StanModel(file='stan/fir_tc.stan')
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
    else:
        print("invalid prior, options are 'hs' or 'tc' ")

    fit = model.sampling(data=stan_data, init=init_function, iter=6000, chains=4)

    traces = fit.extract()

    return (fit, traces)
