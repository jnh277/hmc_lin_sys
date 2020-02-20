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
from helpers import plot_trace



# specific data path
data_path = 'data/c_tanks3.mat'
data = loadmat(data_path)

y_est = data['y_estimation']
u_est = data['u_estimation'].flatten()
y_val = data['y_validation']
u_val = data['u_validation'].flatten()
Ts = data['Ts'].flatten()
states = data['states_est']

no_obs_est = np.shape(y_est)[1]
no_obs_val = np.shape(y_val)[1]


# Run Stan
def init_function():
    output = dict(r=1.0,
                  q=1.0,
                  Cq=[1.0,1.0,1.0],
                  Rq=[1.0,1.0,1.0,1.0],
                  )
    return output

model = pystan.StanModel(file='stan/c_tanks3.stan')

stan_data = {'no_obs_est': np.shape(y_est)[1],
             'no_obs_val': np.shape(y_val)[1],
             'y_est': y_est,
             'u_est':u_est,
             'y_val':y_val,
             'u_val':u_val,
             'Ts':Ts[0],
             }

control = {"adapt_delta": 0.8,
           "max_treedepth":10}         # increasing from default 0.8 to reduce divergent steps

fit = model.sampling(data=stan_data, init=init_function, iter=5000, chains=4,control=control)

# print(fit)

traces = fit.extract()
yhat = traces['y_hat']
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0

yhat_mean = np.mean(yhat, axis=0)
yhat_upper_ci = np.percentile(yhat, 97.5, axis=0)
yhat_lower_ci = np.percentile(yhat, 2.5, axis=0)


# Cq_traces = np.exp(traces['log_Cq'])
# Rq_traces = np.exp(traces['log_Rq'])
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

h_upper_ci = np.percentile(h_traces, 97.5, axis=0)
h_lower_ci = np.percentile(h_traces, 2.5, axis=0)

plt.subplot(1,2,1)
plt.plot(y_est[0,:],linewidth=0.5)
plt.plot(yhat_mean[0,:],linewidth=0.5)
plt.plot(yhat_upper_ci[0,:],'--',linewidth=0.5)
plt.plot(yhat_lower_ci[0,:],'--',linewidth=0.5)
plt.title('Y1 estimation')
plt.legend(('measurements','mean','upper CI','lower CI'))
plt.subplot(1,2,2)
plt.plot(y_est[1,:],linewidth=0.5)
plt.plot(yhat_mean[1,:],linewidth=0.5)
plt.plot(yhat_upper_ci[1,:],'--',linewidth=0.5)
plt.plot(yhat_lower_ci[1,:],'--',linewidth=0.5)
plt.title('Y2 estimation')
plt.legend(('measurements','mean','upper CI','lower CI'))
plt.show()

plt.subplot(1,3,1)
plt.plot(states[0,:],linewidth=0.5)
plt.plot(h_mean[0,:],linewidth=0.5)
plt.plot(h_upper_ci[0,:],'--',linewidth=0.5)
plt.plot(h_lower_ci[0,:],'--',linewidth=0.5)
plt.title('state 1 estimation')
plt.legend(('true','mean','upper CI','lower CI'))
plt.subplot(1,3,2)
plt.plot(states[1,:],linewidth=0.5)
plt.plot(h_mean[1,:],linewidth=0.5)
plt.plot(h_upper_ci[1,:],'--',linewidth=0.5)
plt.plot(h_lower_ci[1,:],'--',linewidth=0.5)
plt.title('state 2 estimation')
plt.legend(('true','mean','upper CI','lower CI'))
plt.subplot(1,3,3)
plt.plot(states[2,:],linewidth=0.5)
plt.plot(h_mean[2,:],linewidth=0.5)
plt.plot(h_upper_ci[2,:],'--',linewidth=0.5)
plt.plot(h_lower_ci[2,:],'--',linewidth=0.5)
plt.title('state 3 estimation')
plt.legend(('true','mean','upper CI','lower CI'))
plt.show()



plot_trace(Cq_traces[:,0],4,1,'Cq 1')
plot_trace(Rq_traces[:,0],4,2,'Rq 1')
plot_trace(r_traces,4,3,'r')
plot_trace(q_traces,4,4,'q')
plt.show()

plt.subplot(1,1,1)
plt.plot(Cq_traces[:,1],Rq_traces[:,1],'o')
plt.show()

## validation
states_val = data["states_val"]




