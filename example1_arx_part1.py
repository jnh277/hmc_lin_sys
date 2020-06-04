###############################################################################
#    Practical Bayesian System Identification using Hamiltonian Monte Carlo
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

""" runs the code for example 1 part 1 in the paper and produces the figures"""

import numpy as np
from helpers import calculate_acf
import matplotlib.pyplot as plt
from helpers import plot_dbode_ML
import seaborn as sns
from scipy.io import loadmat

from arx_hmc import run_arx_hmc
from arx_mh import run_arx_mh



# specific data path
data_path = 'data/arx_example_part_one.mat'


# specify model orders, not nb = 3 as opposed to 2 in the paper, because numbering starts at one not zero
input_order = 3         # gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}
output_order = 2        # gives the terms a_0 * y_{k-1} + ... + a_{output_order-1}*y_{k-output_order}

# load validation data with which to calculate the model fit metric
data = loadmat(data_path)
y_val = data['y_validation'].flatten()
max_delay = np.max((output_order, input_order - 1))
y_val = y_val[int(max_delay):]

a_ML = data['a_ML'][0,1:]

# estimate using hmc with horeshoe prior
(fit_hmc,results_hmc) = run_arx_hmc(data_path, input_order, output_order, prior='l2')

a_hmc = results_hmc['a_coefs']
b_hmc = results_hmc['b_coefs']
a_coef_mean = np.mean(a_hmc,0)
b_coef_mean = np.mean(b_hmc,0)
a1_hmc = a_hmc[:,0]
acf_hmc = calculate_acf(a1_hmc)

yhat_hmc = results_hmc['y_hat']      # validation predictions
yhat_hmc[np.isnan(yhat_hmc)] = 0.0
yhat_hmc[np.isinf(yhat_hmc)] = 0.0
yhat_mean_hmc = np.mean(yhat_hmc, axis=0)


# calculate MF
MF_hmc = 100*(1-np.sum(np.power(y_val-yhat_mean_hmc,2))/np.sum(np.power(y_val,2)))

# estimate using metropolis hastings
results_mh = run_arx_mh(mh2=False)
a1_mh = results_mh['a1']
acf_mh = calculate_acf(a1_mh)

# estimate using mMALA
results_mMala = run_arx_mh(mh2=True)
a1_mMALA = results_mMala['a1']
acf_mMALA = calculate_acf(a1_mMALA)

# check goodness of fit, and plot some diagnostics
fontsize= 16

plt.subplot(2,2,1)
plt.plot(np.arange(750,825),a1_hmc[750:825])
plt.plot(np.arange(750,825),a1_mh[750:825])
plt.plot(np.arange(750,825),a1_mMALA[750:825])
plt.ylabel('$a_1$',fontsize=fontsize)
plt.xlabel('iteration', fontsize=fontsize)

plt.subplot(2,2,2)
sns.kdeplot(a1_hmc, shade=True)
sns.kdeplot(a1_mh, shade=True)
sns.kdeplot(a1_mMALA, shade=True)
plt.axvline(a_ML[0], color='k', lw=1, linestyle='--')
plt.xlabel('$a_1$', fontsize=fontsize)
plt.ylabel('posterior', fontsize=fontsize)

plt.subplot(2,2,3)
plt.plot(acf_hmc)
plt.plot(acf_mh)
plt.plot(acf_mMALA)
plt.xlabel('lag', fontsize=fontsize)
plt.ylabel('ACF of $a_1$', fontsize=fontsize)
# plt.legend(('chain 1','chain 2','chain 3','chain 4'), fontsize=20)
plt.tight_layout()
plt.savefig("figures/example1_diagnostics.png", format='png')
plt.show()






plt.subplot(1,1,1)
plt.plot(y_val,'+',linewidth=0.5)
plt.plot(yhat_mean_hmc,linewidth=0.5)
plt.xlabel('t',fontsize=fontsize)
plt.ylabel('y',fontsize=fontsize)
plt.title('One step ahead prediction for validaiton data')
plt.legend(('Measurement','mean prediction'))
plt.show()





# do a pairs plot to look at marginal and joint posterior distributions
fontsize = 24

plt.figure(figsize=(10,8.25))
plt.subplot(3,3,1)
plt.hist(a_hmc[:,1],30, density=True)
sns.kdeplot(a_hmc[:,1], shade=True)
plt.axvline(np.mean(a_hmc[:,1]), color='k', lw=1, linestyle='--', label='mean')
plt.xlabel('$a_2$', fontsize=fontsize)
plt.ylabel('marginal', fontsize=fontsize)

plt.subplot(3,3,4)
plt.plot(a_hmc[:,1],b_hmc[:,0],'.')
plt.xlabel('$a_2$',fontsize=fontsize)
plt.ylabel('$b_0$',fontsize=fontsize)

plt.subplot(3,3,5)
plt.hist(b_hmc[:,0],30, density=True)
sns.kdeplot(b_hmc[:,0], shade=True)
plt.axvline(np.mean(b_hmc[:,0]), color='k', lw=1, linestyle='--', label='mean')
plt.xlabel('$b_0$', fontsize=fontsize)
plt.ylabel('marginal', fontsize=fontsize)

plt.subplot(3,3,7)
plt.plot(a_hmc[:,1],b_hmc[:,1],'.')
plt.xlabel('$a_2$',fontsize=fontsize)
plt.ylabel('$b_1$',fontsize=fontsize)

plt.subplot(3,3,8)
plt.plot(b_hmc[:,0],b_hmc[:,1],'.')
plt.xlabel('$b_0$',fontsize=fontsize)
plt.ylabel('$b_1$',fontsize=fontsize)

plt.subplot(3,3,9)
plt.hist(b_hmc[:,1],30, density=True)
sns.kdeplot(b_hmc[:,1], shade=True)
plt.axvline(np.mean(b_hmc[:,1]), color='k', lw=1, linestyle='--', label='mean')
plt.xlabel('$b_1$', fontsize=fontsize)
plt.ylabel('marginal', fontsize=fontsize)

plt.tight_layout()
# plt.savefig('figures/example_1_dists.png',format='png')
plt.show()


b_true = data["b_true"]
a_true = data["a_true"]
Ts = 1.0

w_res = 100
w_plot = np.logspace(-2,np.log10(3.14),w_res)

a_ML = data['a_ML']
b_ML = data['b_ML']

plot_dbode_ML(b_hmc,a_hmc,b_true,a_true,b_ML,a_ML,Ts,w_plot)

