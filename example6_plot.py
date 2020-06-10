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

""" Loads saved results for example 6 and plots """

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from helpers import plot_trace
import pickle
import scipy.io as sio

# load data
data_path = 'data/wheeled_robot_sysid_data.mat'
data = loadmat(data_path)

theta = data['h'].flatten()
u1 = data['u1'].flatten()
u2 = data['u2'].flatten()
x = data['x'].flatten()
y = data['y'].flatten()

no_obs = len(y)

with open('data/wheeled_robot_results.pickle','rb') as file:
    traces = pickle.load(file)

mass = traces['m']
length = traces['l']
inertia = traces['J']
z = traces['h']

mass_mean = np.mean(mass, 0)
length_mean = np.mean(length, 0)
inertia_mean = np.mean(inertia, 0)
z_mean = np.mean(z, 0)
z_upper_ci = np.percentile(z, 99.9, axis=0)
z_lower_ci = np.percentile(z, 0.1, axis=0)


LQ = traces['LQ']
LQ_mean = np.mean(LQ, 0)
LR = traces['LR']
LR_mean = np.mean(LR, 0)

R = np.matmul(LR_mean, LR_mean.T)
Q = np.matmul(LQ_mean, LQ_mean.T)


parameter_traces = dict(mass=mass,length=length,inertia=inertia,LQ=LQ,LR=LR)
sio.savemat('rover_parameter_traces.mat', parameter_traces)


plot_trace(mass, 3, 1, 'mass')
plot_trace(length, 3, 2, 'length')
plot_trace(inertia, 3, 3, 'inertia')
plt.show()

plt.subplot(2, 2, 1)
plt.plot(x, y)
plt.plot(z_mean[0, :], z_mean[1, :], '--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Measurements','mean estimate'])
plt.subplot(2, 2, 2)
plt.plot(u1)
plt.ylabel('u1')
plt.xlabel('t')
plt.subplot(2, 2, 3)
plt.plot(u2)
plt.ylabel('u2')
plt.xlabel('t')
plt.subplot(2, 2, 4)
plt.plot(theta)
plt.plot(z_mean[2, :], '--')
plt.legend(['Measurements','mean estimate'])
plt.ylabel('heading (theta)')
plt.xlabel('t')
plt.show()

plt.subplot(3,3,1)
plt.hist(z[:,0,15],30)
plt.xlabel('x at $t_{15}$')
plt.ylabel('counts')

plt.subplot(3,3,2)
plt.hist(z[:,0,200],30)
plt.xlabel('x at $t_{200}$')
plt.ylabel('counts')

plt.subplot(3,3,3)
plt.hist(z[:,0,400],30)
plt.xlabel('x at $t_{400}$')
plt.ylabel('counts')

plt.subplot(3,3,4)
plt.hist(z[:,1,15],30)
plt.xlabel('y at $t_{15}$')
plt.ylabel('counts')

plt.subplot(3,3,5)
plt.hist(z[:,1,200],30)
plt.xlabel('y at $t_{200}$')
plt.ylabel('counts')

plt.subplot(3,3,6)
plt.hist(z[:,1,400],30)
plt.xlabel('y at $t_{400}$')
plt.ylabel('counts')

plt.subplot(3,3,7)
plt.hist(z[:,2,15],30)
plt.xlabel('h at $t_{15}$')
plt.ylabel('counts')

plt.subplot(3,3,8)
plt.hist(z[:,2,200],30)
plt.xlabel('h at $t_{200}$')
plt.ylabel('counts')

plt.subplot(3,3,9)
plt.hist(z[:,2,400],30)
plt.xlabel('h at $t_{400}$')
plt.ylabel('counts')
plt.show()


