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

import stan
from scipy.io import loadmat
import pickle
from pathlib import Path


def run_lssm_hmc(data_path, number_states, hot_start=False, iter=2000, discrete=False):
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
                          D=float(data['D_ML'][0, 0]),
                          h=data['x_ML']
                          )
            return output
    else:
        def init_function():
            output = dict(r=1.0,
                          D=data['D_ML'][0, 0],
                          )
            return output

    if discrete:
        f = open('stan/lssm_discrete.stan', 'r')
        model_code = f.read()
    else:
        f = open('stan/ssm_horseshoe.stan', 'r')
        model_code = f.read()

    stan_data = {'no_obs_est': len(y_est),
                 'y_est': y_est,
                 'u_est': u_est,
                 'Ts': Ts[0],
                 'no_states': number_states,
                 }

    # control = {"adapt_delta": 0.8,
    #            "max_treedepth": 10}  # increasing from default 0.8 to reduce divergent steps

    posterior = stan.build(model_code, data=stan_data)
    init = [init_function(),init_function(),init_function(),init_function()]
    traces = posterior.sample(init=init, num_samples=iter, num_warmup=4000, num_chains=4)
# , max_depth=10, adapt_delt=0.8
    return traces
