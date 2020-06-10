"""Estimates the system from mcha6100 assignment 2and its states."""
import pystan
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from helpers import plot_trace
import pickle
import scipy.io as sio


# load data
data_path = 'data/ass2SysidData.mat'
data = loadmat(data_path)

theta = data['h'].flatten()
u1 = data['u1'].flatten()
u2 = data['u2'].flatten()
x = data['x'].flatten()
y = data['y'].flatten()

no_obs = len(y)

with open('/data/wheeled_robot_results.pickle','rb') as file:
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


# plt.subplot(3,1,1)
# plt.plot(x,'r.',MarkerSize=2)
# plt.plot(z_mean[0,:],'b')
# plt.plot(z_upper_ci[0,:],'b--')
# plt.plot(z_lower_ci[0,:],'b--')
# plt.xlabel('t_k')
# plt.ylabel('x')
# plt.legend(['Measurements','Mean','Upper CI','Lower CI'])
#
# plt.subplot(3,1,2)
# plt.plot(y,'r.',MarkerSize=2)
# plt.plot(z_mean[1,:],'b')
# plt.plot(z_upper_ci[1,:],'b--')
# plt.plot(z_lower_ci[1,:],'b--')
# plt.xlabel('t_k')
# plt.ylabel('y')
# plt.legend(['Measurements','Mean','Upper CI','Lower CI'])
#
# plt.subplot(3,1,3)
# plt.plot(theta,'r.',MarkerSize=2)
# plt.plot(z_mean[1,:],'b')
# plt.plot(z_upper_ci[1,:],'b--')
# plt.plot(z_lower_ci[1,:],'b--')
# plt.xlabel('t_k')
# plt.ylabel('y')
# plt.legend(['Measurements','Mean','Upper CI','Lower CI'])
#
# plt.show()
