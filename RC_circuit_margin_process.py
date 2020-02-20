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
"""Estimates a RC circuit and its state."""


import pystan
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import plot_trace


# specific data path
data_path = 'data/rc_circuit.mat'
data = loadmat(data_path)



y_est = data['y_estimation'].flatten()
u_est = data['u_estimation'].flatten()
y_val = data['y_validation'].flatten()
u_val = data['u_validation'].flatten()
Ts = data['Ts'].flatten()

# reduce number of measurements to try this out
N = 100
y_est = y_est[:N]
u_est = u_est[:N]

no_obs_est = len(y_est)
no_obs_val = len(y_val)


# Run Stan
def init_function():
    output = dict(r=1.0,
                  q=1.0,
                  Cq=1.0,
                  Rq=1.0,
                  )
    return output

model = pystan.StanModel(file='stan/RC_circuit_margin_process.stan')

stan_data = {'no_obs_est': len(y_est),
             'no_obs_val': len(y_val),
             'y_est': y_est,
             'u_est':u_est,
             'Ts':Ts[0],
             }

control = {"adapt_delta": 0.8,
           "max_treedepth":10}         # increasing from default 0.8 to reduce divergent steps

fit = model.sampling(data=stan_data, init=init_function, iter=2000, chains=4,control=control)

# print(fit)

traces = fit.extract()
# yhat = traces['y_hat']
# yhat[np.isnan(yhat)] = 0.0
# yhat[np.isinf(yhat)] = 0.0
#
# yhat_mean = np.mean(yhat, axis=0)
# yhat_upper_ci = np.percentile(yhat, 97.5, axis=0)
# yhat_lower_ci = np.percentile(yhat, 2.5, axis=0)


Cq_traces = traces['Cq']
Rq_traces = traces['Rq']
q_traces = traces['q']
r_traces = traces['r']
h_traces = traces['h']


Cq_mean = np.mean(Cq_traces,0)
Rq_mean = np.mean(Rq_traces,0)
h_mean = np.mean(h_traces,0)
r_mean = np.mean(r_traces,0)
q_mean = np.mean(q_traces,0)



# plt.subplot(1,1,1)
# plt.plot(y_est,linewidth=0.5)
# plt.plot(yhat_mean,linewidth=0.5)
# plt.plot(yhat_upper_ci,'--',linewidth=0.5)
# plt.plot(yhat_lower_ci,'--',linewidth=0.5)
# plt.show()



plot_trace(Cq_traces,4,1,'Cq')
plot_trace(Rq_traces,4,2,'Rq')
plot_trace(q_traces,4,3,'q')
plot_trace(r_traces,4,4,'r')
plt.show()


plt.subplot(1,1,1)
plt.plot(Cq_traces,Rq_traces,'o')
plt.xlabel('Cq')
plt.ylabel('Rq')
plt.title('sample pairs plot')
plt.show()

