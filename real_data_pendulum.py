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
# # data_path = 'data/pendulum_data0.mat'
# data = loadmat(data_path)
#
# Ts = data['dt']
# mu0 = data['mu0']
# theta0 = data['theta0']
# u = data['u']
# y = data['y']

data_path ='data/pendulum_data_all_sets.mat'
data = loadmat(data_path)
set_number = 1
Ts = data['dt']
# theta0 = data['theta_init'][:,0]
u = data['u_all'][set_number,:,:]
y = data['y_all'][set_number,:,:]


theta0 = np.ones((6))

no_obs = len(y[0])


# known parameters
Lr = 0.085      # arm length
Mp = 0.025      # pendulum mass
Lp = 0.129      # pendulum length
g  = 9.81       # gravity

# state initialisation point
z_init = np.zeros((4,no_obs))
z_init[0,:] = y[0,:]
z_init[1,:] = y[1,:]
z_init[2,:-1] = (y[0,1:]-y[0,0:-1])/Ts
z_init[2,-1] = z_init[2,-2]
z_init[3,:-1] = (y[1,1:]-y[1,0:-1])/Ts
z_init[3,-1] = z_init[3,-2]

model = pystan.StanModel(file='stan/pendulum.stan')

stan_data = {'no_obs': no_obs,
             'Ts':Ts[0,0],
             'y': y,
             'u': u.flatten(),
             'Lr':Lr,
             'Mp':Mp,
             'Lp':Lp,
             'g':g,
             # 'z0':mu0.flatten(),
             }

control = {"adapt_delta": 0.85,
           "max_treedepth":13}         # increasing from default 0.8 to reduce divergent steps

def init_function():
    output = dict(theta=theta0.flatten() * np.random.uniform(0.8,1.2,np.shape(theta0.flatten())),
                  h=z_init + np.random.normal(0.0,0.1,np.shape(z_init)),
                  )
    return output


fit = model.sampling(data=stan_data, iter=5000, chains=4,control=control, init=init_function)
# fit = model.sampling(data=stan_data, iter=10, chains=1,control=control, init=init_function)


traces = fit.extract()

with open('results/pendulum_set1_results.pickle', 'wb') as file:
    pickle.dump(traces, file)


theta = traces['theta']
z = traces['h']

theta_mean = np.mean(theta,0)
z_mean = np.mean(z,0)

# LQ = traces['LQ']
# LQ_mean = np.mean(LQ,0)
# LR = traces['LR']
# LR_mean = np.mean(LR,0)
#
# R = np.matmul(LR_mean, LR_mean.T)
# Q = np.matmul(LQ_mean, LQ_mean.T)

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

