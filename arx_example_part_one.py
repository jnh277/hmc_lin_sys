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

"""Estimates an ARX model using data with Gaussian noise and known model orders."""
""" A horseshoe prior is placed on the coefficients a_{1:n_a} and b_{1:n_b} """

import pystan
import numpy as np
from scipy.io import loadmat
from helpers import build_input_matrix
from helpers import build_obs_matrix
from helpers import calculate_acf
import matplotlib.pyplot as plt
from helpers import plot_dbode
from helpers import plot_dbode_ML
import seaborn as sns

# specific data path

data_path = 'data/arx_example_part_one.mat'

# specify model orders, not nb = 3 as opposed to 2 in the paper, because numbering starts at one not zero
input_order = 3         # gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}
output_order = 2        # gives the terms a_0 * y_{k-1} + ... + a_{output_order-1}*y_{k-output_order}

data = loadmat(data_path)

y_est = data['y_estimation'].flatten()
u_est = data['u_estimation'].flatten()
y_val = data['y_validation'].flatten()
u_val = data['u_validation'].flatten()

no_obs_est = len(y_est)
no_obs_val = len(y_val)

# build regression matrix
est_input_matrix = build_input_matrix(u_est, input_order)
est_obs_matrix = build_obs_matrix(y_est, output_order)
val_input_matrix = build_input_matrix(u_val, input_order)
val_obs_matrix = build_obs_matrix(y_val, output_order)

# trim measurement vectors to suit regression matrix
max_delay = np.max((output_order,input_order-1))
y_est = y_est[int(max_delay):]
y_val = y_val[int(max_delay):]


# Set up parameter initialisation, initialise from +/- 40% of the maximum likelihood estimate
def init_function():
    a_init = data['a_ML'].flatten()[1:output_order+1]
    b_init = data['b_ML'].flatten()
    sig_e_init = data['sig_e_ML'].flatten()
    output = dict(a_coefs=a_init * np.random.uniform(0.6, 1.4, len(a_init)),
                  b_coefs=b_init * np.random.uniform(0.6, 1.4, len(b_init)),
                  sig_e=(sig_e_init * np.random.uniform(0.6, 1.4))[0],
                  shrinkage_param=np.abs(np.random.standard_cauchy(1))[0]
                  )
    return output

# specify model file
model = pystan.StanModel(file='stan/arx.stan')

# specify the data
stan_data = {'input_order': int(input_order),
             'output_order': int(output_order),
             'no_obs_est': len(y_est),
             'no_obs_val': len(y_val),
             'y_est': y_est,
             'est_obs_matrix': est_obs_matrix,
             'est_input_matrix': est_input_matrix,
             'val_obs_matrix': val_obs_matrix,
             'val_input_matrix': val_input_matrix
             }

# perform sampling using hamiltonian monte carlo
fit = model.sampling(data=stan_data, init=init_function, iter=6000, chains=4)


# extract the results
traces = fit.extract()


# extract parameter samples
a_coef_traces = traces['a_coefs']
b_coef_traces = traces['b_coefs']
shrinkage_param = traces["shrinkage_param"]
shrinkage_param_mean = np.mean(shrinkage_param,0)

a_coef_mean = np.mean(a_coef_traces,0)
b_coef_mean = np.mean(b_coef_traces,0)




# check goodness of fit, and plot some diagnostics
fontsize= 16

a_1_chain1 = a_coef_traces[0:3000,0]
a_1_chain2 = a_coef_traces[3000:6000,0]
a_1_chain3 = a_coef_traces[6000:9000,0]
a_1_chain4 = a_coef_traces[9000:,0]

plt.subplot(2,2,1)
plt.plot(np.arange(750,825),a_1_chain1[750:825])
plt.plot(np.arange(750,825),a_1_chain2[750:825])
plt.plot(np.arange(750,825),a_1_chain2[750:825])
plt.plot(np.arange(750,825),a_1_chain3[750:825])
plt.ylabel('$a_1$',fontsize=fontsize)
plt.xlabel('iteration', fontsize=fontsize)

plt.subplot(2,2,2)
# plt.hist(a_1_chain1,30)
sns.kdeplot(a_1_chain1, shade=True)
sns.kdeplot(a_1_chain2, shade=True)
sns.kdeplot(a_1_chain3, shade=True)
sns.kdeplot(a_1_chain4, shade=True)
plt.xlabel('$a_1$', fontsize=fontsize)
plt.ylabel('posterior', fontsize=fontsize)

acf1 = calculate_acf(a_1_chain1)
acf2 = calculate_acf(a_1_chain2)
acf3 = calculate_acf(a_1_chain3)
acf4 = calculate_acf(a_1_chain4)

plt.subplot(2,2,3)
plt.plot(acf1)
plt.plot(acf2)
plt.plot(acf3)
plt.plot(acf4)
plt.xlabel('lag', fontsize=fontsize)
plt.ylabel('ACF of $a_1$', fontsize=fontsize)
# plt.legend(('chain 1','chain 2','chain 3','chain 4'), fontsize=20)
plt.tight_layout()
# plt.savefig("figures/example1_diagnostics.png", format='png')
plt.show()


# extract validation data predictions
yhat = traces['y_hat']      # validation predictions
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0
yhat_mean = np.mean(yhat, axis=0)
yhat_upper_ci = np.percentile(yhat, 99.5, axis=0)
yhat_lower_ci = np.percentile(yhat, 0.5, axis=0)


plt.subplot(1,1,1)
plt.plot(y_val,'+',linewidth=0.5)
plt.plot(yhat_mean,linewidth=0.5)
plt.xlabel('t',fontsize=fontsize)
plt.ylabel('y',fontsize=fontsize)
plt.title('One step ahead prediction for validaiton data')
plt.legend(('Measurement','mean prediction'))
plt.show()

# calculate MF
MF = 100*(1-np.sum(np.power(y_val-yhat_mean,2))/np.sum(np.power(y_val,2)))



# do a pairs plot to look at marginal and joint posterior distributions
fontsize = 24

plt.figure(figsize=(10,8.25))
plt.subplot(3,3,1)
plt.hist(a_coef_traces[:,1],30, density=True)
sns.kdeplot(a_coef_traces[:,1], shade=True)
plt.axvline(np.mean(a_coef_traces[:,1]), color='k', lw=1, linestyle='--', label='mean')
plt.xlabel('$a_2$', fontsize=fontsize)
plt.ylabel('marginal', fontsize=fontsize)

plt.subplot(3,3,4)
plt.plot(a_coef_traces[:,1],b_coef_traces[:,0],'.')
plt.xlabel('$a_2$',fontsize=fontsize)
plt.ylabel('$b_0$',fontsize=fontsize)

plt.subplot(3,3,5)
plt.hist(b_coef_traces[:,0],30, density=True)
sns.kdeplot(b_coef_traces[:,0], shade=True)
plt.axvline(np.mean(b_coef_traces[:,0]), color='k', lw=1, linestyle='--', label='mean')
plt.xlabel('$b_0$', fontsize=fontsize)
plt.ylabel('marginal', fontsize=fontsize)

plt.subplot(3,3,7)
plt.plot(a_coef_traces[:,1],b_coef_traces[:,1],'.')
plt.xlabel('$a_2$',fontsize=fontsize)
plt.ylabel('$b_1$',fontsize=fontsize)

plt.subplot(3,3,8)
plt.plot(b_coef_traces[:,0],b_coef_traces[:,1],'.')
plt.xlabel('$b_0$',fontsize=fontsize)
plt.ylabel('$b_1$',fontsize=fontsize)

plt.subplot(3,3,9)
plt.hist(b_coef_traces[:,1],30, density=True)
sns.kdeplot(b_coef_traces[:,1], shade=True)
plt.axvline(np.mean(b_coef_traces[:,1]), color='k', lw=1, linestyle='--', label='mean')
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

plot_dbode_ML(b_coef_traces,a_coef_traces,b_true,a_true,b_ML,a_ML,Ts,w_plot)

