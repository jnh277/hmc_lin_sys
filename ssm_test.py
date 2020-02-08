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

"""Estimates a state space system of order 1 and simultaneously the hidden states."""
import pystan
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns


# specific data path
data_path = 'data/ss_order1.mat'
data = loadmat(data_path)

y_est = data['y_estimation']
u_est = data['u_estimation']
y_val = data['y_validation']
u_val = data['u_validation']

y_est = y_est[1:500]
u_est = u_est[1:500]

no_obs_est = len(y_est)
no_obs_val = len(y_val)

no_states = 1
no_outputs = 1
Ts = 0.1

# Run Stan
def init_function():
    output = dict(L_Q=np.diag(np.ones((no_states,1)).flatten()),
                  L_R=np.diag(np.ones((no_outputs,1)).flatten()),
                  )
    return output

model = pystan.StanModel(file='stan/ssm_uc.stan')

stan_data = {'no_obs_est': len(y_est),
             'no_obs_val': len(y_val),
             'y_est': y_est,
             'u_est':u_est,
             'y_val':y_val,
             'u_val':u_val,
             'no_states':1,
             'no_inputs':1,
             'no_outputs':1,
             'Ts':Ts
             }

control = {"adapt_delta": 0.8,
           "max_treedepth":10}         # increasing from default 0.8 to reduce divergent steps

fit = model.sampling(data=stan_data,init=init_function, iter=5000, chains=4,control=control,n_jobs=1)

# print(fit)

traces = fit.extract()
yhat = traces['y_hat']
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0

yhat_mean = np.mean(yhat, axis=0)
yhat_upper_ci = np.percentile(yhat, 97.5, axis=0)
yhat_lower_ci = np.percentile(yhat, 2.5, axis=0)


a_traces = traces['A']
b_traces = traces['B']
c_traces = traces['C']
d_traces = traces['D']
q_traces = traces['Q']
r_traces = traces['R']
h_traces = traces['h']


a_mean = np.mean(a_traces,0)
b_mean = np.mean(b_traces,0)
c_mean = np.mean(c_traces,0)
d_mean = np.mean(d_traces,0)
h_mean = np.mean(h_traces,0)
r_mean = np.mean(r_traces,0)
q_mean = np.mean(q_traces,0)
# d_hyper_mean = np.mean(d_hyper_traces,0)


plt.subplot(1,1,1)
plt.plot(y_est,linewidth=0.5)
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
    plt.hist(param, 30, density=True)
    sns.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.axvline(mean, color='r', lw=2, linestyle='--', label='mean')
    plt.axvline(median, color='c', lw=2, linestyle='--', label='median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()

plot_trace(a_traces,4,1,'a')
plot_trace(b_traces,4,2,'b')
plot_trace(c_traces,4,3,'c')
plot_trace(d_traces,4,4,'d')
plt.show()


