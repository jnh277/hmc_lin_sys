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

""" Runs the code for example 6 in the paper and produces the figures """
""" This demonstrates Bayesian estimation of non-linear state space models using HMC """
""" Since this takes a long time to run it also saves the results so that they can be """
""" loaded and plotted at a later stage """


import pystan
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from helpers import plot_trace
import pickle


# load data
data_path = 'data/wheeled_robot_sysid_data.mat'
data = loadmat(data_path)

theta = data['h'].flatten()
u1 = data['u1'].flatten()
u2 = data['u2'].flatten()
x = data['x'].flatten()
y = data['y'].flatten()


no_obs = len(y)

Y = np.zeros((3,no_obs))
Y[0,:] = x
Y[1,:] = y
Y[2,:] = theta

Ts = 0.1
r1 = 1
r2 = 1
a = 0.5
z0 = np.array([x[0],y[0],theta[0],0,0])
theta_diff = theta[1:]-theta[:-1]
z_init = [x,y,theta,0*x,0*x]


model = pystan.StanModel(file='stan/wheeled_robot.stan')

stan_data = {'no_obs': no_obs,
             'Ts':Ts,
             'y': Y,
             'u1': u1,
             'u2':u2,
             'a':a,
             'r1':r1,
             'r2':r2,
             'z0':z0,
             }

control = {"adapt_delta": 0.85,
           "max_treedepth":13}         # increasing from default 0.8 to reduce divergent steps

def init_function():
    output = dict(m = 5 * np.random.uniform(0.5,1.5),
                  J = 2 * np.random.uniform(0.5,1.5),
                  # phi = 1/2* np.random.uniform(0.8,1.2),
                  l = 0.15 * np.random.uniform(0.5,1.5),
                  h = z_init + np.random.normal(0.0,0.4,np.shape(z_init)),
                  )
    return output

fit = model.sampling(data=stan_data, iter=5000, chains=4,control=control, init=init_function)
# fit = model.sampling(data=stan_data, iter=10, chains=1,control=control, init=init_function)


traces = fit.extract()

with open('results/wheeled_robot_results.pickle', 'wb') as file:
    pickle.dump(traces, file)

# to read
# with open('rover_results.pickle') as file:
#     traces = pickle.load(file)


mass = traces['m']
length = traces['l']
inertia = traces['J']
z = traces['h']

mass_mean = np.mean(mass,0)
length_mean = np.mean(length,0)
inertia_mean = np.mean(inertia,0)
z_mean = np.mean(z,0)

LQ = traces['LQ']
LQ_mean = np.mean(LQ,0)
LR = traces['LR']
LR_mean = np.mean(LR,0)

R = np.matmul(LR_mean, LR_mean.T)
Q = np.matmul(LQ_mean, LQ_mean.T)

plot_trace(mass,3,1,'mass')
plot_trace(length,3,2,'length')
plot_trace(inertia,3,3,'inertia')
plt.show()

plt.subplot(2,2,1)
plt.plot(x,y)
plt.plot(z_mean[0,:],z_mean[1,:],'--')
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(2,2,2)
plt.plot(u1)
plt.ylabel('u1')
plt.xlabel('t')
plt.subplot(2,2,3)
plt.plot(u2)
plt.ylabel('u2')
plt.xlabel('t')
plt.subplot(2,2,4)
plt.plot(theta)
plt.plot(z_mean[2,:],'--')
plt.plot(np.percentile(z[:,2,:],97.5,axis=0),'--')
plt.plot(np.percentile(z[:,2,:],2.5,axis=0),'--')
plt.ylabel('heading (theta)')
plt.xlabel('t')
plt.show()

