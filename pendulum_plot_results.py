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


import pystan
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from helpers import plot_trace
import pickle



# load data
data_path = 'data/pendulum_data1.mat'
data = loadmat(data_path)

Ts = data['dt']
mu0 = data['mu0']
theta0 = data['theta0']
u = data['u']
y = data['y']

no_obs = len(y[0])

# state initialisation point
z_init = np.zeros((4,no_obs))
z_init[0,:] = y[0,:]
z_init[1,:] = y[1,:]
z_init[2,:-1] = (y[0,1:]-y[0,0:-1])/Ts
z_init[2,-1] = z_init[2,-2]
z_init[3,:-1] = (y[1,1:]-y[1,0:-1])/Ts
z_init[3,-1] = z_init[3,-2]



# with open('results/pendulum_data1_trial0.pickle','rb') as file:
#     traces = pickle.load(file)
with open('results/pendulum_results_ones_init.pickle','rb') as file:
    traces = pickle.load(file)

theta = traces['theta']
z = traces['h']
yhat = traces['yhat']

lp = traces['lp__']

theta_mean = np.mean(theta,0)
z_mean = np.mean(z,0)

LQ = traces['LQ']
LQ_mean = np.mean(LQ,0)
LR = traces['LR']
LR_mean = np.mean(LR,0)

R = np.matmul(LR_mean, LR_mean.T)
Q = np.matmul(LQ_mean, LQ_mean.T)

print('mean theta = ', theta_mean)

plot_trace(theta[:,0],3,1,'Jr')
plot_trace(theta[:,1],3,2,'Jp')
plot_trace(theta[:,2],3,3,'Km')
plt.show()

plot_trace(theta[:,3],3,1,'Rm')
plot_trace(theta[:,4],3,2,'Dp')
plot_trace(theta[:,5],3,3,'Dr')
plt.show()

plt.subplot(2,2,1)
plt.plot(y[0,:])
plt.plot(z_mean[0,:])
plt.xlabel('time')
plt.ylabel(r'arm angle $\theta$')
plt.legend(['Measurements','mean estimate'])

plt.subplot(2,2,2)
plt.plot(y[1,:])
plt.plot(z_mean[1,:])
plt.xlabel('time')
plt.ylabel(r'pendulum angle $\alpha$')
plt.legend(['Measurements','mean estimate'])

plt.subplot(2,2,3)
plt.plot(z_init[2,:])
plt.plot(z_mean[2,:])
plt.xlabel('time')
plt.ylabel(r'arm angular velocity $\dot{\theta}$')
plt.legend(['Grad measurements','mean estimate'])

plt.subplot(2,2,4)
plt.plot(z_init[3,:])
plt.plot(z_mean[3,:])
plt.xlabel('time')
plt.ylabel(r'pendulum angular velocity $\dot{\alpha}$')
plt.legend(['Grad measurements','mean estimate'])
plt.show()

plt.hist(lp,100)
plt.xlabel('sample log-posterior')
plt.ylabel('count')
plt.show()

# plt.subplot(2,2,1)
# plt.plot(yhat[:,2,49],z[:,2,49],'.')
# plt.xlabel(r'$\hat{y}_3$ at t=50')
# plt.ylabel(r'$\dot{\theta}$ at t=50')
#
# plt.subplot(2,2,2)
# plt.plot(yhat[:,2,49],z[:,3,49],'.')
# plt.xlabel(r'$\hat{y}_2$ at t=50')
# plt.ylabel(r'$\dot{\alpha}$ at t=50')
#
#
# plt.subplot(2,2,3)
# plt.plot(yhat[:,2,99],z[:,2,99],'.')
# plt.xlabel(r'$\hat{y}_3$ at t=100')
# plt.ylabel(r'$\dot{\theta}$ at t=100')
#
# plt.subplot(2,2,4)
# plt.plot(yhat[:,2,99],z[:,3,99],'.')
# plt.xlabel(r'$\hat{y}_2$ at t=100')
# plt.ylabel(r'$\dot{\alpha}$ at t=100')
# plt.show()
