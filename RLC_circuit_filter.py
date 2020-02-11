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

"""Estimates a RLC circuit and its states."""
import pystan
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path


# specific data path
data_path = 'data/rlc_circuit.mat'
data = loadmat(data_path)

y_est = data['y_estimation'].flatten()
u_est = data['u_estimation'].flatten()
y_val = data['y_validation'].flatten()
u_val = data['u_validation'].flatten()
Ts = data['Ts'].flatten()
Cq = data['Cq'].flatten()
Rq = data['Rq'].flatten()
Lq = data['Lq'].flatten()
Q = data['Q'].flatten()
R = data['R'].flatten()
states = data['states_est']

no_obs_est = len(y_est)
no_obs_val = len(y_val)


# Run Stan
# def init_function():
#     output = dict(r=1.0,
#                   # q=1.0,
#                   Cq=1.0,
#                   Rq=1.0,
#                   Lq = 1.0,
#                   )
#     return output

## ------ now use STAN to get a bayesian estimate ------------
save_file = Path('stan/RLC_circuit_filter.pkl')
if save_file.is_file():
    model = pickle.load(open('stan/RLC_circuit_filter.pkl', 'rb'))
else:
    # compile stan model
    model = pystan.StanModel(file="stan/RLC_circuit_filter.stan")
    # save compiled file
    # save it to the file 'trunc_normal_model.pkl' for later use
    with open('stan/RLC_circuit_filter.pkl', 'wb') as f:
        pickle.dump(model, f)

# model = pystan.StanModel(file='stan/RLC_circuit_filter.stan')

stan_data = {'no_obs_est': len(y_est),
             'no_obs_val': len(y_val),
             'y_est': y_est,
             'u_est':u_est,
             'y_val':y_val,
             'u_val':u_val,
             'Ts':Ts[0],
             'Cq':Cq[0],
             'Rq':Rq[0],
             'Lq':Lq[0],
             'q':np.sqrt(Q)[0],
             'r':np.sqrt(R)[0],
             }

control = {"adapt_delta": 0.8,
           "max_treedepth":10}         # increasing from default 0.8 to reduce divergent steps

fit = model.sampling(data=stan_data, iter=5000, chains=4,control=control)

# print(fit)

traces = fit.extract()
yhat = traces['y_hat']
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0

yhat_mean = np.mean(yhat, axis=0)
yhat_upper_ci = np.percentile(yhat, 97.5, axis=0)
yhat_lower_ci = np.percentile(yhat, 2.5, axis=0)


h_traces = traces["h"]
h_traces_mean = np.mean(h_traces,0)


# Cq_traces = np.exp(traces['log_Cq'])
# Rq_traces = np.exp(traces['log_Rq'])

# q_mean = np.mean(q_traces,0)



plt.subplot(1,1,1)
plt.plot(y_est,linewidth=0.5)
plt.plot(yhat_mean,linewidth=0.5)
plt.plot(yhat_upper_ci,'--',linewidth=0.5)
plt.plot(yhat_lower_ci,'--',linewidth=0.5)
plt.title('measurement estimation')
plt.show()

h1 = h_traces[:,0,:]

plt.subplot(1,1,1)
plt.plot(states[0,:],linewidth=0.5)
plt.plot(h_traces_mean[0,:],linewidth=0.5)
plt.plot(np.percentile(h1, 97.5, axis=0),'--',linewidth=0.5)
plt.plot(np.percentile(h1, 2.5, axis=0),'--',linewidth=0.5)
plt.title('state 1 estimation')
plt.show()

h2 = h_traces[:,1,:]

plt.subplot(1,1,1)
plt.plot(states[1,:],linewidth=0.5)
plt.plot(h_traces_mean[1,:],linewidth=0.5)
plt.plot(np.percentile(h2, 97.5, axis=0),'--',linewidth=0.5)
plt.plot(np.percentile(h2, 2.5, axis=0),'--',linewidth=0.5)
plt.title('state 2 estimation')
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



