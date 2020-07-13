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

""" Runs the code for example 2 in the paper and produces the figures """
""" This demonstrates Bayesian estiamtion of FIR models using HMC """

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from fir_hmc import run_fir_hmc
import seaborn as sns

# specific data path
data_path = 'data/example2_fir.mat'
input_order = 35       # gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}

# load validation data for computing model fit metric
data = loadmat(data_path)
y_val = data['y_validation'].flatten()
max_delay = (input_order-1)
y_val = y_val[int(max_delay):]


## estime using horseshoe prior
(fit, traces) = run_fir_hmc(data_path, input_order,  prior='hs', hot_start=True)


##
yhat_hs = traces['y_hat']
yhat_hs[np.isnan(yhat_hs)] = 0.0
yhat_hs[np.isinf(yhat_hs)] = 0.0

yhat_mean_hs = np.mean(yhat_hs, axis=0)
yhat_upper_ci_hs = np.percentile(yhat_hs, 97.5, axis=0)
yhat_lower_ci_hs = np.percentile(yhat_hs, 2.5, axis=0)


b_hs = traces['b_coefs']
b_hs_mean = np.mean(b_hs,0)


MF_hs = 100*(1-np.sum(np.power(y_val-yhat_mean_hs,2))/np.sum(np.power(y_val,2)))

## estime using TC prior
(fit, traces) = run_fir_hmc(data_path, input_order,  prior='tc', hot_start=True)


##
yhat_tc = traces['y_hat']
yhat_tc[np.isnan(yhat_tc)] = 0.0
yhat_tc[np.isinf(yhat_tc)] = 0.0

yhat_mean_tc = np.mean(yhat_tc, axis=0)
yhat_upper_ci_tc = np.percentile(yhat_tc, 99.5, axis=0)
yhat_lower_ci_tc = np.percentile(yhat_tc, 0.5, axis=0)


b_tc = traces['b_coefs']
b_tc_mean = np.mean(b_tc,0)


MF_tc = 100*(1-np.sum(np.power(y_val-yhat_mean_tc,2))/np.sum(np.power(y_val,2)))

## PLOT results

# plot 1 step ahead predictions
yhat_arx_TC = data['y_hat_val_ML']
yhat_arx_TC = yhat_arx_TC[int(max_delay):]

inds = np.arange(50,120)
plt.subplot(1,1,1)
plt.plot(inds,y_val[inds],'k',linewidth=1)
plt.plot(inds,yhat_mean_tc[inds],linewidth=1)
plt.fill_between(inds,yhat_lower_ci_tc[inds],yhat_upper_ci_tc[inds])
plt.plot(inds,yhat_arx_TC[inds],'--',linewidth=1)
plt.ylabel('one-step-ahead prediction',fontsize=16)
plt.xlabel('t',fontsize=16)
plt.legend(('y','hmc TC mean','ARX TC','hmc TC 99% interval'))
plt.savefig('figures/fir_onestep.png',format='png')
plt.show()


# plot_trace(b_tc[:,0],4,1,'b[0]')
# plot_trace(b_tc[:,1],4,2,'b[1]')
# plot_trace(b_tc[:,11],4,3,'b[11]')
# plot_trace(b_tc[:,12],4,4,'b[12]')
# plt.show()
# Ts = 1.0
# num_true = data['b_true']
# den_true = data['a_true']
# plot_firfreq(b_tc,num_true,den_true,b_ML.flatten())

#



## plot parameter values
b_ML0 = data['b_ML0']       # arx without TC
b_ML = data['b_ML']         # arx with TC

plt.subplot(2,2,2)
# plt.plot(np.arange(0,len(np.mean(b_hs,axis=0))),np.mean(b_hs,axis=0))
sns.boxplot(data=b_hs,fliersize=0.01)
plt.xticks([0,10,20,30],["0","10","20","30"])
plt.xlabel('$b_k$',fontsize=16)
plt.title('HMC with horseshow prior')

plt.subplot(2,2,4)
sns.boxplot(data=b_tc,fliersize=0.01)
plt.xticks([0,10,20,30],["0","10","20","30"])
plt.xlabel('$b_k$',fontsize=16)
plt.title('HMC with TC prior')
plt.ylim([-0.02,0.23])

plt.subplot(2,2,1)
plt.plot(b_ML0[0,:],'--s')
plt.xticks([0,5,10])
plt.xlabel('$b_k$',fontsize=16)
plt.title('ARX assuming $n_b=13$')

plt.subplot(2,2,3)
plt.plot(b_ML[0,:],'--s')
plt.xlabel('$b_k$',fontsize=16)
plt.title('ARX with TC kernel')
plt.ylim([-0.02,0.23])
plt.tight_layout()
plt.savefig('figures/fir_params.png',format='png')
plt.show()

g_true = data['g_true']

## plot impulse response
upper = np.percentile(b_hs, 97.5, axis=0)
lower = np.percentile(b_hs, 2.5, axis=0)
plt.subplot(2,2,2)
plt.plot(g_true,'k')
plt.plot(np.arange(0,len(np.mean(b_hs,axis=0))),np.mean(b_hs,axis=0))
plt.fill_between(np.arange(0,len(np.mean(b_hs,axis=0))),lower,upper,alpha=0.2)
plt.xlabel('$t$',fontsize=16)
plt.ylabel('impulse response',fontsize=12)
# plt.legend(('true','HMC with horseshow prior','95% CI'))
plt.title('HMC with horseshoe prior')

upper = np.percentile(b_tc, 97.5, axis=0)
lower = np.percentile(b_tc, 2.5, axis=0)
plt.subplot(2,2,4)
plt.plot(g_true,'k')
plt.plot(np.arange(0,len(np.mean(b_tc,axis=0))),np.mean(b_tc,axis=0))
plt.fill_between(np.arange(0,len(np.mean(b_hs,axis=0))),lower,upper,alpha=0.2)
plt.xlabel('$t$',fontsize=16)
plt.ylabel('impulse response',fontsize=12)
# plt.legend(('true','HMC with TC prior'))
plt.ylim([-0.02,0.23])
plt.title('HMC with TC prior')

plt.subplot(2,2,1)
plt.plot(g_true,'k')
plt.plot(b_ML0[0,:])
plt.xlabel('$t$',fontsize=16)
plt.ylabel('impulse response',fontsize=12)
# plt.legend(('True','ARX assuming $n_b=13$'))
plt.title('ML assuming $n_b = 13$')

plt.subplot(2,2,3)
plt.plot(g_true,'k')
plt.plot(b_ML[0,:])
plt.xlabel('$t$',fontsize=16)
plt.ylabel('impulse response',fontsize=12)
# plt.legend(('true','ARX with TC kernel','95% CI'))
plt.ylim([-0.02,0.23])
plt.title('ML with TC regularisation')
plt.tight_layout()
plt.savefig('fir_impulse.png',format='png')
plt.show()

# compute and plot step response
step_true = data['step_true']

hmc_hs_step = np.zeros(np.shape(b_hs))
hmc_tc_step = np.zeros(np.shape(b_tc))
ML_step = np.zeros(np.shape(b_tc_mean))
ML_tc_step = np.zeros(np.shape(b_tc_mean))
for k in range(len(b_hs_mean)):
    if k==0:
        hmc_hs_step[:, k] = b_hs[:, (k + 1)]
        hmc_tc_step[:, k] = b_tc[:, (k + 1)]
        ML_step[k] = np.sum(b_ML0[0,:(k + 1)])
        ML_tc_step[k] = np.sum(b_ML[0,:(k + 1)])
    else:
        hmc_hs_step[:,k] = np.sum(b_hs[:,:(k+1)],axis=1)
        hmc_tc_step[:,k] = np.sum(b_tc[:,:(k + 1)], axis=1)
        ML_step[k] = np.sum(b_ML0[0,:(k + 1)])
        ML_tc_step[k] = np.sum(b_ML[0,:(k + 1)])

upper = np.percentile(hmc_hs_step, 97.5, axis=0)
lower = np.percentile(hmc_hs_step, 2.5, axis=0)
plt.subplot(2,2,2)
plt.plot(step_true,'k')
plt.plot(np.mean(hmc_hs_step,axis=0))
plt.fill_between(np.arange(0,len(np.mean(b_hs,axis=0))),lower,upper,alpha=0.2)
plt.legend(('True','hmc horseshoe prior','95% CI'))
plt.xlabel('t',fontsize=16)
plt.ylabel('step response',fontsize=12)

upper = np.percentile(hmc_tc_step, 97.5, axis=0)
lower = np.percentile(hmc_tc_step, 2.5, axis=0)
plt.subplot(2,2,4)
plt.plot(step_true,'k')
plt.plot(np.mean(hmc_tc_step,axis=0))
plt.fill_between(np.arange(0,len(np.mean(b_hs,axis=0))),lower,upper,alpha=0.2)
plt.legend(('True','hmc TC prior','95% CI'))
plt.xlabel('t',fontsize=16)
plt.ylabel('step response',fontsize=12)

plt.subplot(2,2,1)
plt.plot(step_true,'k')
plt.plot(ML_step)
plt.legend(('True','arx without TC'))
plt.xlabel('t',fontsize=16)
plt.ylabel('step response',fontsize=12)

plt.subplot(2,2,3)
plt.plot(step_true,'k')
plt.plot(ML_tc_step)
plt.legend(('True','arx with TC'))
plt.xlabel('t',fontsize=16)
plt.ylabel('step response',fontsize=12)
plt.savefig('figures/fir_step.png',format='png')
plt.show()