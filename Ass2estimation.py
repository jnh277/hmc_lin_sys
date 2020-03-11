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

"""Estimates the system from mcha6100 assignment 2and its states."""
import pystan
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from helpers import plot_trace


# load data
data_path = 'data/ass2SysidData.mat'
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

plt.subplot(2,2,1)
plt.plot(x,y)
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
plt.ylabel('heading (theta)')
plt.xlabel('t')
plt.show()


model = pystan.StanModel(file='stan/ass2.stan')

stan_data = {'no_obs': no_obs,
             'Ts':Ts,
             'y': Y,
             'u1': u1,
             'u2':u2,
             'a':a,
             'r1':r1,
             'r2':r2,
             'z0':z0
             }

control = {"adapt_delta": 0.8,
           "max_treedepth":10}         # increasing from default 0.8 to reduce divergent steps

fit = model.sampling(data=stan_data, iter=2000, chains=4,control=control)