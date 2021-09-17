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

""" Runs the code for example 1 part 2 (Section 6.2) in the paper and produces the figures """
""" This compares L1, L2, and Horseshoe priors for ARX models in HMC """


import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from helpers import plot_dbode_ML, plot_d_nyquist
import seaborn as sns
import pandas as pd
import platform
if platform.system()=='Darwin':
    import multiprocessing
    multiprocessing.set_start_method("fork")

from arx_hmc import run_arx_hmc

# specific data path

data_path = 'data/arx_example_part_two_e2_reg.mat'
# specify model orders, note nb = 11 as opposed to 10 in the paper, because numbering starts at one not zero (total number of terms is correct)
input_order = 11         # gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}
output_order = 10        # gives the terms a_0 * y_{k-1} + ... + a_{output_order-1}*y_{k-output_order}


data = loadmat(data_path)

# load the validation data for computing model fit metric
y_val = data['y_validation'].flatten()
max_delay = np.max((output_order,input_order-1))
y_val = y_val[int(max_delay):]

# fit using hmc with horseshoe prior
(fit_hs,traces) = run_arx_hmc(data_path, input_order, output_order, prior='hs', hot_start=False)

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

# fit using hmc with horseshoe prior
(fit_l2,traces) = run_arx_hmc(data_path, input_order, output_order, prior='l2', hot_start=False)


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

# fit using hmc with horseshoe prior
(fit_l1,traces) = run_arx_hmc(data_path, input_order, output_order, prior='l1', hot_start=False)


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
MF_ML = data['MF_ML']

## plot the results

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
b_true = data["b_true"]
a_true = data["a_true"]
a_ML = data["a_ML"]
b_ML = data["b_ML"]


Ts = 1.0
w_res = 500
w_plot = np.logspace(-2,np.log10(3.14),w_res)
plot_dbode_ML(b_hs,a_hs,b_true,a_true,b_ML,a_ML,Ts,w_plot, save=True)

# function for creating the nyquist plot of a discrete time system
from scipy import signal



plot_d_nyquist(b_hs,a_hs,b_true,a_true,b_ML,a_ML,Ts,w_plot, save='figures/arx_nyquist.png')