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
import matplotlib.pyplot as plt
from helpers import plot_trace
from helpers import plot_firfreq


# specific data path
data_path = 'data/example2_fir.mat'
input_order = 35       # gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}

# load validation data for computing model fit metric
data = loadmat(data_path)
y_val = data['y_validation'].flatten()
max_delay = (input_order-1)
y_val = y_val[int(max_delay):]

## estimation goes here



##
yhat = traces['y_hat']
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0

yhat_mean = np.mean(yhat, axis=0)
yhat_upper_ci = np.percentile(yhat, 97.5, axis=0)
yhat_lower_ci = np.percentile(yhat, 2.5, axis=0)

stan_est = {'mean': yhat_mean, 'upper': yhat_upper_ci, 'lower': yhat_lower_ci,
            'sig_e_mean':traces['sig_e'].mean()}

b_hyper_traces = traces['b_coefs_hyperprior']
b_coef_traces = traces['b_coefs']
shrinkage_param = traces["shrinkage_param"]
shrinkage_param_mean = np.mean(shrinkage_param,0)

b_hyper_mean = np.mean(b_hyper_traces,0)
b_coef_mean = np.mean(b_coef_traces,0)

plt.subplot(1,1,1)
plt.plot(y_val,linewidth=0.5)
plt.plot(yhat_mean,linewidth=0.5)
plt.plot(yhat_upper_ci,'--',linewidth=0.5)
plt.plot(yhat_lower_ci,'--',linewidth=0.5)
plt.show()



plot_trace(b_coef_traces[:,0],4,1,'b[0]')
plot_trace(b_coef_traces[:,1],4,2,'b[1]')
plot_trace(b_coef_traces[:,11],4,3,'b[11]')
plot_trace(b_coef_traces[:,12],4,4,'b[12]')
plt.show()
Ts = 1.0

b_ML = data['b_ML']
num_true = data['b_true']
den_true = data['a_true']
plot_firfreq(b_coef_traces,num_true,den_true,b_ML.flatten())

plt.subplot(2,1,1)
plt.plot(np.mean(b_coef_traces,axis=0))
plt.plot(b_ML[0,:])
plt.show()
