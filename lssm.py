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

"""Estimates parameters and states of a SISO lssm using a horseshoe prior on the parameters"""

import pystan
from scipy.io import loadmat
import pickle
from pathlib import Path

def run_lssm_hmc(data_path, number_states, hot_start=False, iter=4000,discrete=False):
    data = loadmat(data_path)
    y_est = data['y_estimation'].flatten()
    u_est = data['u_estimation'].flatten()
    Ts = data['Ts'].flatten()

    # Run Stan
    if hot_start:
        def init_function():
            output = dict(r=1.0,
                          A=data['A_ML'],
                          B=data['B_ML'].flatten(),
                          C=data['C_ML'].flatten(),
                          D=data['D_ML'][0,0],
                          h=data['x_ML']
                          )
            return output
    else:
        def init_function():
            output = dict(r=1.0,
                          D=data['D_ML'][0,0],
                          )
            return output

    if discrete:
        model_path = 'stan/lssm_hs_discrete.pkl'
        if Path(model_path).is_file():
            model = pickle.load(open(model_path, 'rb'))
        else:
            model = pystan.StanModel(file='stan/lssm_discrete.stan')
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
    else:
        model_path = 'stan/lssm_hs.pkl'
        if Path(model_path).is_file():
            model = pickle.load(open(model_path, 'rb'))
        else:
            model = pystan.StanModel(file='stan/ssm_horseshoe.stan')
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)


    stan_data = {'no_obs_est': len(y_est),
                 'y_est': y_est,
                 'u_est':u_est,
                 'Ts':Ts[0],
                 'no_states':number_states,
                 }

    control = {"adapt_delta": 0.8,
               "max_treedepth":10}         # increasing from default 0.8 to reduce divergent steps

    fit = model.sampling(data=stan_data, init=init_function, iter=iter, chains=4,control=control)

    traces = fit.extract()
    return (fit, traces)
