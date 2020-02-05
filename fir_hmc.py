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
import pandas as pd
from scipy.io import loadmat
from helpers import build_input_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# specific data path
data_path = 'data/fir_order2.mat'
input_order = 20       # gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}

data = loadmat(data_path)

y_est = data['y_estimation'].flatten()
u_est = data['u_estimation'].flatten()
y_val = data['y_validation'].flatten()
u_val = data['u_validation'].flatten()

no_obs_est = len(y_est)
no_obs_val = len(y_val)

# build regression matrix
est_input_matrix = build_input_matrix(u_est, input_order)
val_input_matrix = build_input_matrix(u_val, input_order)

# trim measurement vectors to suit regression matrix
max_delay = (input_order-1)
y_est = y_est[int(max_delay):]
y_val = y_val[int(max_delay):]

# calcualte an intial guess using least squares (ML)
Ainv = np.linalg.pinv(est_input_matrix)
b_init = np.matmul(Ainv, y_est)

# Run Stan
def init_function():
    sig_e = data['sig_e'].flatten()
    output = dict(b_coefs=b_init * np.random.uniform(0.8, 1.2, len(b_init)),
                  sig_e=(sig_e * np.random.uniform(0.8, 1.2))[0],
                  b_coefs_hyperprior=np.abs(np.random.standard_cauchy(len(b_init))),
                  shrinkage_param=np.abs(np.random.standard_cauchy(1))[0]
                  )
    return output

model = pystan.StanModel(file='stan/fir.stan')

stan_data = {'input_order': int(input_order),
             'no_obs_est': len(y_est),
             'no_obs_val': len(y_val),
             'y_est': y_est,
             'est_input_matrix': est_input_matrix,
             'val_input_matrix': val_input_matrix
             }
fit = model.sampling(data=stan_data, init=init_function, iter=5000, chains=4)

traces = fit.extract()
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

def plot_trace(param,num_plots,pos, param_name='parameter'):
    """Plot the trace and posterior of a parameter."""

    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

    # Plotting
    plt.subplot(num_plots, 1, pos)
    plt.hist(param, 30, density=True);
    sns.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.axvline(mean, color='r', lw=2, linestyle='--', label='mean')
    plt.axvline(median, color='c', lw=2, linestyle='--', label='median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()


plot_trace(b_coef_traces[:,0],4,1,'b[0]')
plot_trace(b_coef_traces[:,1],4,2,'b[1]')
plot_trace(b_coef_traces[:,14],4,3,'b[14]')
plot_trace(b_coef_traces[:,15],4,4,'b[15]')
plt.show()


