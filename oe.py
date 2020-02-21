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
import matplotlib.pyplot as plt
from helpers import plot_trace
from helpers import plot_dbode
from scipy import signal


# specific data path
data_path = 'data/oe_order2.mat'
input_order = 3
output_order = 2

data = loadmat(data_path)

y_est = data['y_estimation'].flatten()
u_est = data['u_estimation'].flatten()
y_val = data['y_validation'].flatten()
u_val = data['u_validation'].flatten()

no_obs_est = len(y_est)
no_obs_val = len(y_val)


# Run Stan
def init_function():
    f_true = data['f_coef_true'].flatten()[1:output_order+1]
    b_true = data['b_coef_true'].flatten()
    sig_e = data['sig_e'].flatten()
    output = dict(f_coefs=np.flip(f_true) * np.random.uniform(0.8, 1.2, len(f_true)),
                  b_coefs=np.flip(b_true)* np.random.uniform(0.8, 1.2, len(b_true)),
                  # r = 1.0,
                  r=(sig_e * np.random.uniform(0.8, 1.2))[0],
                  # a_coefs_hyperprior=np.abs(np.random.standard_cauchy(len(f_true))),
                  # b_coefs_hyperprior=np.abs(np.random.standard_cauchy(len(b_true))),
                  # shrinkage_param=np.abs(np.random.standard_cauchy(1))[0]
                  )
    return output

model = pystan.StanModel(file='stan/oe.stan')

stan_data = {'input_order': int(input_order),
             'output_order': int(output_order),
             'no_obs_est': len(y_est),
             'y_est': y_est,
             'y_val': y_val,
             'u_est':u_est,
             'u_val':u_val,
             'no_obs_val': len(y_val),
             }
# stan_data = {'order_b': int(input_order),
#              'order_f': int(output_order),
#              'no_obs_est': len(y_est),
#              'y_est': y_est,
#              'y_val': y_val,
#              'u_est':u_est,
#              'u_val':u_val,
#              'no_obs_val': len(y_val),
#              }

fit = model.sampling(data=stan_data, init=init_function, iter=2000, chains=4)

traces = fit.extract()
yhat = traces['y_hat_val']
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0

yhat_mean = np.mean(yhat, axis=0)
yhat_upper_ci = np.percentile(yhat, 97.5, axis=0)
yhat_lower_ci = np.percentile(yhat, 2.5, axis=0)


# mu_traces = traces['mu']
f_coef_traces = traces['f_coefs']
b_coef_traces = traces['b_coefs']
# shrinkage_param = traces["shrinkage_param"]
# shrinkage_param_mean = np.mean(shrinkage_param,0)
# r_traces = traces["r"]

# mu_mean = np.mean(mu_traces,0)
# r_mean = np.mean(r_traces,0)

f_mean = np.mean(f_coef_traces,0)
b_mean = np.mean(b_coef_traces,0)

plt.subplot(1,1,1)
plt.plot(y_val,linewidth=0.5)
plt.plot(yhat_mean,linewidth=0.5)
plt.plot(yhat_upper_ci,'--',linewidth=0.5)
plt.plot(yhat_lower_ci,'--',linewidth=0.5)
plt.ylim((-2,2))
plt.show()



plot_trace(f_coef_traces[:,0],4,1,'f[0]')
plot_trace(f_coef_traces[:,1],4,2,'f[2]')
plot_trace(b_coef_traces[:,0],4,3,'b[0]')
plot_trace(b_coef_traces[:,1],4,4,'b[1]')
plt.show()


# now plot the bode diagram

f_true = data["f_coef_true"]
b_true = data["b_coef_true"]

Ts = data["Ts"]
w_res = 100
w_plot = np.logspace(-3,1,w_res)

plot_dbode(b_coef_traces[:,-1::-1],f_coef_traces[:,-1::-1],b_true.flatten(),f_true.flatten(),Ts,w_plot)
