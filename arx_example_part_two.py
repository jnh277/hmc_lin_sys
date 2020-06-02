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

"""Estimates an ARX model using data with Gaussian noise and unknown model orders."""
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
import pandas as pd

# specific data path

data_path = 'data/arx_example_part_two_e2_reg.mat'
# specify model orders, not nb = 11 as opposed to 10 in the paper, because numbering starts at one not zero
input_order = 11         # gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}
output_order = 10        # gives the terms a_0 * y_{k-1} + ... + a_{output_order-1}*y_{k-output_order}


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
# def init_function():
#     a_init = data['a_ML'].flatten()[1:output_order+1]
#     b_init = data['b_ML'].flatten()
#     sig_e_init = data['sig_e_ML'].flatten()
#     output = dict(a_coefs=np.concatenate((np.zeros(1),a_init * np.random.uniform(0.8, 1.2, len(a_init))),0),
#                   b_coefs=np.concatenate((np.zeros(3),b_init * np.random.uniform(0.8, 1.2, len(b_init))),0),
#                   sig_e=(sig_e_init * np.random.uniform(0.8, 1.2))[0],
#                   shrinkage_param=np.abs(np.random.standard_cauchy(1))[0]
#                   )
#     return output
def init_function():
    a_init = data['a_ML'].flatten()[1:output_order+1]
    b_init = data['b_ML'].flatten()
    sig_e_init = data['sig_e_ML'].flatten()
    output = dict(a_coefs=a_init * np.random.uniform(0.8, 1.2, len(a_init)),
                  b_coefs=b_init * np.random.uniform(0.8, 1.2, len(b_init)),
                  sig_e=(sig_e_init * np.random.uniform(0.8, 1.2))[0],
                  shrinkage_param=np.abs(np.random.standard_cauchy(1))[0]
                  )
    return output


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

## fit using a horseshoe prior
# specify model file
model_hs = pystan.StanModel(file='stan/arx.stan')

# perform sampling using hamiltonian monte carlo
fit_hs = model_hs.sampling(data=stan_data, init=init_function, iter=6000, chains=4)


# extract the results
traces = fit_hs.extract()


# extract parameter samples
a_hs= traces['a_coefs']
b_hs = traces['b_coefs']
a_hs_mean = np.mean(a_hs,0)
b_hs_mean = np.mean(b_hs,0)
yhat = traces['y_hat']      # validation predictions
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0
yhat_mean = np.mean(yhat, axis=0)

MF_hs = 100*(1-np.sum(np.power(y_val-yhat_mean,2))/np.sum(np.power(y_val,2)))

## fit using an L2 prior
# specify model file
model_l2 = pystan.StanModel(file='stan/arx_l2.stan')


# perform sampling using hamiltonian monte carlo
fit_l2 = model_l2.sampling(data=stan_data, init=init_function, iter=6000, chains=4)


# extract the results
traces = fit_l2.extract()


# extract parameter samples
a_l2= traces['a_coefs']
b_l2 = traces['b_coefs']
a_l2_mean = np.mean(a_l2,0)
b_l2_mean = np.mean(b_l2,0)
yhat = traces['y_hat']      # validation predictions
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0
yhat_mean = np.mean(yhat, axis=0)

MF_l2 = 100*(1-np.sum(np.power(y_val-yhat_mean,2))/np.sum(np.power(y_val,2)))

## fit using an L1 prior
# specify model file
model_l1 = pystan.StanModel(file='stan/arx_l1.stan')


# perform sampling using hamiltonian monte carlo
fit_l1 = model_l1.sampling(data=stan_data, init=init_function, iter=6000, chains=4)


# extract the results
traces = fit_l1.extract()


# extract parameter samples
a_l1= traces['a_coefs']
b_l1 = traces['b_coefs']
a_l1_mean = np.mean(a_l1,0)
b_l1_mean = np.mean(b_l1,0)
yhat = traces['y_hat']      # validation predictions
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0
yhat_mean = np.mean(yhat, axis=0)

MF_l1 = 100*(1-np.sum(np.power(y_val-yhat_mean,2))/np.sum(np.power(y_val,2)))

a_ML = data['a_ML'][0,1:]
b_ML = data['b_ML'][0,:]

## plots
fontsize = 16
plt.subplot(2,1,1)
plt.plot(np.arange(1,3),data['a_true'][0,1:],'o')
plt.plot(np.arange(1,11),a_hs_mean,'+')
plt.plot(np.arange(1,11),a_l2_mean,'*')
plt.plot(np.arange(1,11),a_l1_mean,'x')
plt.plot(np.arange(1,len(a_ML)+1),a_ML,'s')
plt.ylabel('$a_k$',fontsize=fontsize)
plt.xlabel('k',fontsize=fontsize)
plt.legend(('True','horseshoe prior','L2 prior','L1','arx'))
plt.grid()

plt.subplot(2,1,2)
plt.plot(np.arange(0,3),data['b_true'][0,:],'o')
plt.plot(np.arange(0,11),b_hs_mean,'+')
plt.plot(np.arange(0,11),b_l2_mean,'*')
plt.plot(np.arange(0,11),b_l2_mean,'x')
plt.plot(np.arange(0,len(b_ML)),b_ML,'s')
plt.ylabel('$b_k$',fontsize=fontsize)
plt.xlabel('k',fontsize=fontsize)
plt.legend(('True','horseshoe prior','L2 prior','L1 prior','arx'))
plt.grid()

plt.show()


## plots
fontsize = 16
plt.subplot(2,1,1)
plt.plot(np.arange(1,9),a_hs_mean[2:],'+')
plt.plot(np.arange(1,9),a_l2_mean[2:],'*')
plt.plot(np.arange(1,9),a_l1_mean[2:],'x')
plt.plot(np.arange(1,len(a_ML)-1),a_ML[2:],'s')
plt.ylabel('$a_k$',fontsize=fontsize)
plt.xlabel('k',fontsize=fontsize)
plt.legend(('True','horseshoe prior','L2 prior','L1','arx'))
plt.grid()

plt.subplot(2,1,2)
plt.plot(np.arange(0,8),b_hs_mean[3:],'+')
plt.plot(np.arange(0,8),b_l2_mean[3:],'*')
plt.plot(np.arange(0,8),b_l2_mean[3:],'x')
plt.plot(np.arange(0,len(b_ML)-3),b_ML[3:],'s')
plt.ylabel('$b_k$',fontsize=fontsize)
plt.xlabel('k',fontsize=fontsize)
plt.legend(('horseshoe prior','L2 prior','L1 prior','arx'))
plt.grid()

plt.show()
plt.boxplot(a_hs[:,2:])
plt.boxplot(a_l1[:,2:])
plt.show()

# show box plot results for a
# convert to data frame
(r,c) = np.shape(a_hs)

data1 = list()
for i in range(r):
    for j in range(2,c):
        data1.append([a_hs[i,j],j+1,'hs'])
for i in range(r):
    for j in range(2,c):
        data1.append([a_l1[i,j],j+1,'L1'])
for i in range(r):
    for j in range(2,c):
        data1.append([a_l2[i,j],j+1,'L2'])
for j in range(2,len(a_ML)):
    data1.append([a_ML[j],j+1,'ML'])

df = pd.DataFrame(data1,columns=['val','index','method'])

fig,(ax1) = plt.subplots(1)
sns.boxplot(x='index',y='val',data=df,hue='method',ax=ax1)

for i,artist in enumerate(ax1.artists):
    # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    # artist.set_edgecolor(col)
    # artist.set_facecolor('None')

    # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    # Loop over them here, and use the same colour as above
    # for j in range(i*6,i*6+6):
    for j in range(i * 6+5, i * 6 + 6):
        line = ax1.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)
    if (i==3 or i==7 or i==11 or i==15 or i==19 or i==23 or i==27 or i==31):
        j = i*6+4
        line = ax1.lines[j]
        line._x = np.mean(line._x)
        line._y = np.mean(line._y)
        tmp = line
        line.set_linewidth(3)
        # line.set_marker('o')
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

# sns.despine(offset=10, trim=True)
plt.xlabel('index k',fontsize=20)
plt.ylabel('$a_k$',fontsize=20)
plt.grid()
ax1.set_axisbelow(True)
plt.savefig('figures/unknown_order_a.png',format='png')
plt.show()


# show box plot results for b
# convert to data frame
(r,c) = np.shape(b_hs)

data2 = list()
for i in range(r):
    for j in range(3,c):
        data2.append([b_hs[i,j],j,'hs'])
for i in range(r):
    for j in range(3,c):
        data2.append([b_l1[i,j],j,'L1'])
for i in range(r):
    for j in range(3,c):
        data2.append([b_l2[i,j],j,'L2'])
for j in range(3,len(b_ML)):
    data2.append([b_ML[j],j,'ML'])

df2 = pd.DataFrame(data2,columns=['val','index','method'])

fig1,(ax1) = plt.subplots(1)
sns.boxplot(x='index',y='val',data=df2,hue='method',ax=ax1)

for i,artist in enumerate(ax1.artists):
    # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    # artist.set_edgecolor(col)
    # artist.set_facecolor('None')

    # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    # Loop over them here, and use the same colour as above
    # for j in range(i*6,i*6+6):
    for j in range(i * 6+5, i * 6 + 6):
        line = ax1.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)
    if (i==3 or i==7 or i==11 or i==15 or i==19 or i==23 or i==27 or i==31):
        j = i*6+4
        line = ax1.lines[j]
        line._x = np.mean(line._x)
        line._y = np.mean(line._y)
        tmp = line
        line.set_linewidth(3)
        # line.set_marker('o')
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

# sns.despine(offset=10, trim=True)
plt.xlabel('index k',fontsize=20)
plt.ylabel('$b_k$',fontsize=20)
plt.grid()
ax1.set_axisbelow(True)
plt.savefig('figures/unknown_order_b.png',format='png')
plt.show()



#
# b_true = data["b_true"]
# a_true = data["a_true"]
# a_ML = data["a_ML"]
# b_ML = data["b_ML"]
#
# # b_ML = np.array([0.051689781466163,   1.016509059425918,   0.497674739448588,  -0.015555723404122,   0.028557782752386,   0.021349077335907,0.027884769733234,  -0.051274773001460])
# # a_ML = np.array([1.0, -1.4889,    0.6740,    0.0518,   -0.0126,   -0.0334,    0.0173,    0.0196,   -0.0108])
# Ts = 1.0
# w_res = 100
# w_plot = np.logspace(-2,np.log10(3.14),w_res)
# plot_dbode_ML(b_hs,a_hs,b_true,a_true,b_ML,a_ML,Ts,w_plot)

