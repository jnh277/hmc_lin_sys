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

"""Estimates an ARX model using data with Gaussian noise."""
import numpy as np
from scipy.io import loadmat
from helpers import plot_trace
import matplotlib.pyplot as plt
from helpers import plot_dbode_ML
from arx_hmc import run_arx_hmc


# specific data path
data_path = 'data/example5_outlier.mat'
input_order = 11         # gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}
output_order = 10        # gives the terms a_0 * y_{k-1} + ... + a_{output_order-1}*y_{k-output_order}

data = loadmat(data_path)
y_val = data['y_validation'].flatten()

(fit,traces) = run_arx_hmc(data_path, input_order, output_order,hot_start=True, prior='st',iter=6000)


yhat = traces['y_hat']
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0
# nu = traces['nu']

yhat_mean = np.mean(yhat, axis=0)
yhat_upper_ci = np.percentile(yhat, 97.5, axis=0)
yhat_lower_ci = np.percentile(yhat, 2.5, axis=0)

stan_est = {'mean': yhat_mean, 'upper': yhat_upper_ci, 'lower': yhat_lower_ci,
            'sig_e_mean':traces['sig_e'].mean()}

a_coef_traces = traces['a_coefs']
b_coef_traces = traces['b_coefs']
shrinkage_param = traces["shrinkage_param"]
shrinkage_param_mean = np.mean(shrinkage_param,0)

a_coef_mean = np.mean(a_coef_traces,0)
b_coef_mean = np.mean(b_coef_traces,0)

plt.subplot(1,1,1)
plt.plot(y_val,linewidth=0.5)
plt.plot(yhat_mean,linewidth=0.5)
plt.plot(yhat_upper_ci,'--',linewidth=0.5)
plt.plot(yhat_lower_ci,'--',linewidth=0.5)
plt.show()


plot_trace(a_coef_traces[:,0],4,1,'a[0]')
plot_trace(a_coef_traces[:,1],4,2,'a[2]')
plot_trace(b_coef_traces[:,0],4,3,'b[0]')
plot_trace(b_coef_traces[:,1],4,4,'b[1]')
plt.show()

b_true = data["b_true"]
a_true = data["a_true"]
Ts = 1.0

w_res = 100
w_plot = np.logspace(-2,np.log10(3.14),w_res)
# plot_dbode(b_coef_traces,a_coef_traces,b_true,a_true,Ts,w_plot)

# a_ML = data['a_ML']
# b_ML = data['b_ML']

a_ML = data['a_ML_reg']
b_ML = data['b_ML_reg']

plot_dbode_ML(b_coef_traces,a_coef_traces,b_true,a_true,b_ML,a_ML,Ts,w_plot)

# w, mag_ML, phase_ML = signal.dbode((b_ML.flatten(), a_ML.flatten(), Ts), w_plot)
