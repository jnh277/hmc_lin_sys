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

""" Runs the code for example 6 (Section 6.7) 'Control design with parameter uncertainty
    in the paper and produces the figures. This demonstrates the utility of havaing samples
    from the posterior in order to aid the design of a controller when there is data uncertainty"""
""" This script runs the estimation and saves the estimated model to 'ctrl_sysid_traces.pickle',
    so that it can be plotted using example6_plotsysid.py which also converts the samples from state
     space form to transfer function form for use in the control design in the matlab script 
     matlab/example6_controldesign.m """

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from helpers import plot_trace
from helpers import plot_bode_ML
from lssm import run_lssm_hmc
import stan
import time
# from scipy.io import loadmat
import pickle


hot_start = True
number_states = 6

# specific data path
data_path = 'data/control_example_data.mat'
data = loadmat(data_path)

y_est = data['y_estimation'].flatten()
u_est = data['u_estimation'].flatten()
# states_est = data['states_est']
Ts = data['Ts'].flatten()

no_obs_est = len(y_est)


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
                      D=float(data['D_ML'][0, 0]),
                      )
        return output


f = open('stan/ssm_KF.stan', 'r')
model_code = f.read()

stan_data = {'no_obs_est': len(y_est),
             'y_est': y_est,
             'u_est': u_est,
             'Ts': float(Ts[0]),
             'no_states': number_states,
             'xhat0':np.zeros((6,)),
             'P0':np.eye(number_states)
             }

posterior = stan.build(model_code, data=stan_data)
init = [init_function(),init_function(),init_function(),init_function()]

ts = time.time()
traces = posterior.sample(init=init, num_samples=2000, num_warmup=4000, num_chains=4)
tf = time.time()
print('run time = ', tf-ts)

# with open('results/ctrl_sysid_traces.pickle', 'wb') as file:
#     pickle.dump(traces, file)

# yhat = traces['y_hat'].swapaxes(0, -1)
# yhat[np.isnan(yhat)] = 0.0
# yhat[np.isinf(yhat)] = 0.0
#
# yhat_mean = np.mean(yhat, axis=0)
# yhat_upper_ci = np.percentile(yhat, 97.5, axis=0)
# yhat_lower_ci = np.percentile(yhat, 2.5, axis=0)

A_traces = traces['A'].swapaxes(0, -1).swapaxes(1, 2)
B_traces = traces['B'].swapaxes(0, -1)
C_traces = traces['C'].swapaxes(0, -1)
D_traces = traces['D'].swapaxes(0, -1)
LQ_traces = traces['LQ'].swapaxes(0, -1).swapaxes(1, 2)
r_traces = traces['r'].swapaxes(0, -1)
# h_traces = traces['h'].swapaxes(0, -1).swapaxes(1, 2)

A_mean = np.mean(A_traces, 0)
B_mean = np.mean(B_traces, 0)
C_mean = np.mean(C_traces, 0)
D_mean = np.mean(D_traces, 0)
# h_mean = np.mean(h_traces, 0)
r_mean = np.mean(r_traces, 0)
LQ_mean = np.mean(LQ_traces, 0)

# h_upper_ci = np.percentile(h_traces, 97.5, axis=0)
# h_lower_ci = np.percentile(h_traces, 2.5, axis=0)

# plt.subplot(1, 1, 1)
# plt.plot(y_est, linewidth=0.5)
# plt.plot(yhat_mean, linewidth=0.5)
# plt.plot(yhat_upper_ci, '--', linewidth=0.5)
# plt.plot(yhat_lower_ci, '--', linewidth=0.5)
# plt.title('measurement estimates')
# plt.legend(('true', 'mean', 'upper CI', 'lower CI'))
# plt.show()

plot_trace(A_traces[:, 1, 0], 4, 1, 'A[2,2]')
plot_trace(C_traces[:, 1], 4, 2, 'C[1]')
plot_trace(D_traces[:, 0], 4, 3, 'D')
plot_trace(r_traces[:, 0], 4, 4, 'r')
plt.show()
#


# BODE diagram
# B_mean = np.array([[0],[1]])
A_true = data['a']
B_true = data['b']
C_true = data['c']
D_true = data['d']

w_plot = np.logspace(-2, 3)
#
A_ML = data['A_ML']
B_ML = data['B_ML']
C_ML = data['C_ML']
D_ML = data['D_ML']

plot_bode_ML(A_traces, B_traces, C_traces, D_traces, A_true, B_true, C_true, D_true, A_ML, B_ML, C_ML, D_ML, w_plot,
             save=False)
