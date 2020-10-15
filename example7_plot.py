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

""" Plots the estimation results for example 7 (Section 6.8) in the paper obtained
    using example7_pendulum.py """


import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pickle
from scipy.io import savemat

data_path ='data/pendulum_data_all_sets.mat'
data = loadmat(data_path)
set_number = 1
Ts = data['dt']
# theta0 = data['theta_init'][:,0]
u = data['u_all'][set_number,:,:]
y = data['y_all'][set_number,:,:]

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
with open('results/pendulum_set1_results_coupled_euler.pickle','rb') as file:
    traces = pickle.load(file)

# savemat('results/pendulum_traces_rk4.mat',traces)

theta = traces['theta']
z = traces['h']
yhat = traces['yhat']

# lm = traces['meas_loglikelihood']
# lp = traces['process_loglikelihood']

theta_mean = np.mean(theta,0)
z_mean = np.mean(z,0)

# LQ = traces['LQ']
# LQ_mean = np.mean(LQ,0)
# LR = traces['LR']
# LR_mean = np.mean(LR,0)
#
# R = np.matmul(LR_mean, LR_mean.T)
# Q = np.matmul(LQ_mean, LQ_mean.T)

# L = traces['L']
# L_mean = np.mean(L,0)
# Omega = np.matmul(L_mean, L_mean.T)

print('mean theta = ', theta_mean)


##
fontsize = 16
ax1 = plt.subplot(3,2,1)
plt.hist(theta[:,0],bins=30,density=True)
plt.axvline(np.mean(theta[:,0]),lw=2.5,color='orange',linestyle='--')
plt.xlabel('$J_r$',fontsize=fontsize)
plt.ylabel('$p(J_r | y_{1:T})$',fontsize=fontsize)
plt.yticks([],[])


from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
ax1.xaxis.set_major_formatter(formatter)


ax2 = plt.subplot(3,2,2)
plt.hist(theta[:,1],bins=30,density=True)
plt.axvline(np.mean(theta[:,1]),lw=2.5,color='orange',linestyle='--')
plt.xlabel('$J_p$',fontsize=fontsize)
plt.ylabel('$p(J_p | y_{1:T})$',fontsize=fontsize)
plt.yticks([],[])


formatter2 = ticker.ScalarFormatter(useMathText=True)
formatter2.set_scientific(True)
formatter2.set_powerlimits((-1,1))
ax2.xaxis.set_major_formatter(formatter2)

ax3 = plt.subplot(3,2,3)
plt.hist(theta[:,2],bins=30,density=True)
plt.axvline(np.mean(theta[:,2]),lw=2.5,color='orange',linestyle='--')
plt.xlabel('$k_m$',fontsize=fontsize)
plt.ylabel('$p(k_m | y_{1:T})$',fontsize=fontsize)
plt.yticks([],[])


formatter3 = ticker.ScalarFormatter(useMathText=True)
formatter3.set_scientific(True)
formatter3.set_powerlimits((-1,1))
ax3.xaxis.set_major_formatter(formatter3)


ax4 = plt.subplot(3,2,4)
plt.hist(theta[:,3],bins=30,density=True)
plt.axvline(np.mean(theta[:,3]),lw=2.5,color='orange',linestyle='--')
plt.xlabel('$R_m$',fontsize=fontsize)
plt.ylabel('$p(R_m | y_{1:T})$',fontsize=fontsize)
plt.yticks([],[])

formatter4 = ticker.ScalarFormatter(useMathText=True)
formatter4.set_scientific(True)
formatter4.set_powerlimits((-1,1))
ax4.xaxis.set_major_formatter(formatter4)

ax5 = plt.subplot(3,2,5)
plt.hist(theta[:,4],bins=30,density=True)
plt.axvline(np.mean(theta[:,4]),lw=2.5,color='orange',linestyle='--')
plt.xlabel('$D_p$',fontsize=fontsize)
plt.ylabel('$p(D_p | y_{1:T})$',fontsize=fontsize)
plt.yticks([],[])

formatter5 = ticker.ScalarFormatter(useMathText=True)
formatter5.set_scientific(True)
formatter5.set_powerlimits((-1,1))
ax5.xaxis.set_major_formatter(formatter5)

ax6 = plt.subplot(3,2,6)
plt.hist(theta[:,5],bins=30,density=True)
plt.axvline(np.mean(theta[:,5]),lw=2.5,color='orange',linestyle='--')
plt.xlabel('$D_r$',fontsize=fontsize)
plt.ylabel('$p(D_r | y_{1:T})$',fontsize=fontsize)
plt.yticks([],[])

formatter6 = ticker.ScalarFormatter(useMathText=True)
formatter6.set_scientific(True)
formatter6.set_powerlimits((-1,1))
ax6.xaxis.set_major_formatter(formatter6)

plt.tight_layout()
plt.savefig('figures/pendulum_params.png',format='png')
plt.show()


##
tt = np.linspace(1*Ts[0,0],375*Ts[0,0],375)
plt.subplot(2,2,1)
plt.plot(tt,y[0,:],'k')
plt.plot(tt,z_mean[0,:],'--')
# plt.fill_between(tt,np.percentile(z[:,0],0.5,axis=0),np.percentile(z[:,0],99.5,axis=0))
plt.xlabel('time (s)',fontsize=fontsize)
plt.ylabel(r'$\theta$',fontsize=fontsize)
# plt.legend(['Measured','mean estimate'])
# plt.xlim((1,2))

plt.subplot(2,2,2)
plt.plot(tt,y[1,:],'k')
plt.plot(tt,z_mean[1,:],'--')
plt.xlabel('time (s)',fontsize=fontsize)
plt.ylabel(r'$\alpha$',fontsize=fontsize)
# plt.legend(['Measured','mean estimate'])
# plt.xlim((1,2))

plt.subplot(2,2,3)
plt.plot(tt,z_init[2,:],'k')
plt.plot(tt,z_mean[2,:],'--')
# plt.plot(np.percentile(z[:,2,:],99,axis=0),'--')
# plt.plot(np.percentile(z[:,2,:],1,axis=0),'--')
plt.xlabel('time (s)',fontsize=fontsize)
plt.ylabel(r'$\dot{\theta}$',fontsize=fontsize)
# plt.legend(['gradient of measured','mean estimate'])

plt.subplot(2,2,4)
plt.plot(tt,z_init[3,:],'k')
plt.plot(tt,z_mean[3,:],'--')
plt.xlabel('time (s)',fontsize=fontsize)
plt.ylabel(r'$\dot{\alpha}$',fontsize=fontsize)
# plt.legend(['gradient of measured','mean estimate'])
plt.tight_layout()
plt.savefig('figures/pendulum_states.png',format='png')
plt.show()


plt.subplot(3,3,1)
plt.hist(z[:,3,00],bins=30,density=True)
plt.yticks([],[])
plt.xlabel(r'$\dot{\alpha}$ at $t=0.8$s',fontsize=fontsize)
plt.ylabel('marginal',fontsize=fontsize)

plt.tight_layout()
plt.show()


## pairs plot
fontsize=12
plt.subplot(3,3,1)
plt.hist(z[:,3,245],bins=30,density=True)
plt.yticks([],[])
plt.xticks([],[])
plt.xlabel(r'$\dot{\alpha}$ at $t=1.8$s',fontsize=fontsize)
plt.ylabel('marginal',fontsize=fontsize)

plt.subplot(3,3,4)
plt.plot(z[:,3,245],theta[:,2],'.')
# plt.yticks([],[])
plt.xticks([],[])
plt.xlabel(r'$\dot{\alpha}$ at $t=1.8$s',fontsize=fontsize)
plt.ylabel(r'$k_m$',fontsize=fontsize)

plt.subplot(3,3,5)
plt.hist(theta[:,2],bins=30,density=True)
plt.yticks([],[])
plt.xticks([],[])
# plt.xlim((np.min(theta[:,2]),np.max(theta[:,2])))
plt.xlabel(r'$k_m$',fontsize=fontsize)
plt.ylabel('marginal',fontsize=fontsize)

plt.subplot(3,3,7)
plt.plot(z[:,3,245],yhat[:,2,225],'.')
# plt.yticks([],[])
# plt.xticks([],[])
plt.xlabel(r'$\dot{\alpha}$ at $t=1.8$s',fontsize=fontsize)
plt.ylabel(r'$I_m$ at $t=1.8$s ',fontsize=fontsize)


plt.subplot(3,3,8)
plt.plot(theta[:,2],yhat[:,2,225],'.')
plt.yticks([],[])
# plt.xlim((np.min(theta[:,2]),np.max(theta[:,2])))
# plt.xticks([],[])
plt.xlabel(r'$k_m$',fontsize=fontsize)
plt.ylabel(r'$I_m$ at $t=1.8$s ',fontsize=fontsize)

plt.subplot(3,3,9)
plt.hist(yhat[:,2,225],bins=30,density=True)
# plt.axvline(np.mean(yhat[:,2,225]),lw=2.5,color='k',linestyle='--')
plt.axvline(y[2,225],lw=2.5,color='k',linestyle='--')
plt.yticks([],[])
# plt.xticks([],[])
plt.xlabel(r'$I_m$ at $t=1.8$s ',fontsize=fontsize)
plt.ylabel('marginal',fontsize=fontsize)

plt.tight_layout()
plt.savefig('figures/pendulum_pairs.png',format='png')
plt.show()
#
#
# plt.plot(y[2,:])
# plt.plot(np.mean(yhat[:,2,:],0))
# plt.show()




# plt.subplot(3,3,1)
# plt.hist(z[:,0,15],30)
# plt.xlabel('x at $t_{15}$')
# plt.ylabel('counts')
#
# plt.subplot(3,3,2)
# plt.hist(z[:,0,200],30)
# plt.xlabel('x at $t_{200}$')
# plt.ylabel('counts')
#
# plt.subplot(3,3,3)
# plt.hist(z[:,0,350],30)
# plt.xlabel('x at $t_{400}$')
# plt.ylabel('counts')
#
# plt.subplot(3,3,4)
# plt.hist(z[:,1,15],30)
# plt.xlabel('y at $t_{15}$')
# plt.ylabel('counts')
#
# plt.subplot(3,3,5)
# plt.hist(z[:,1,200],30)
# plt.xlabel('y at $t_{200}$')
# plt.ylabel('counts')
#
# plt.subplot(3,3,6)
# plt.hist(z[:,1,350],30)
# plt.xlabel('y at $t_{400}$')
# plt.ylabel('counts')
#
# plt.subplot(3,3,7)
# plt.hist(z[:,2,15],30)
# plt.xlabel('h at $t_{15}$')
# plt.ylabel('counts')
#
# plt.subplot(3,3,8)
# plt.hist(z[:,2,200],30)
# plt.xlabel('h at $t_{200}$')
# plt.ylabel('counts')
#
# plt.subplot(3,3,9)
# plt.hist(z[:,2,350],30)
# plt.xlabel('h at $t_{400}$')
# plt.ylabel('counts')
# plt.show()
#
