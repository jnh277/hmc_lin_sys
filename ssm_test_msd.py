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

"""Estimates a msd and its states using a general linear ssm model."""
import pystan
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal


# specific data path
data_path = 'data/msd_sumsins.mat'
data = loadmat(data_path)

y_est = data['y_estimation'].flatten()
u_est = data['u_estimation'].flatten()
states_est = data['states_est']
Ts = data['Ts'].flatten()

no_obs_est = len(y_est)


# Run Stan
def init_function():
    output = dict(r=1.0,
                  A=np.array([[1.0, 0.0],[0.0,1.0]]),
                  B=np.array([[0.0],[1.0]]).flatten(),
                  C=np.ones(2),
                  D=0.0,
                  )
    return output

model = pystan.StanModel(file='stan/ssm_horseshoe.stan')

stan_data = {'no_obs_est': len(y_est),
             'y_est': y_est,
             'u_est':u_est,
             'Ts':Ts[0],
             'no_states':2,
             }

control = {"adapt_delta": 0.8,
           "max_treedepth":10}         # increasing from default 0.8 to reduce divergent steps

fit = model.sampling(data=stan_data, init=init_function, iter=2000, chains=4,control=control)

# print(fit)

traces = fit.extract()
yhat = traces['y_hat']
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0

yhat_mean = np.mean(yhat, axis=0)
yhat_upper_ci = np.percentile(yhat, 97.5, axis=0)
yhat_lower_ci = np.percentile(yhat, 2.5, axis=0)


A_traces = traces['A']
B_traces = traces['B']
C_traces = traces['C']
D_traces = traces['D']
LQ_traces = traces['LQ']
r_traces = traces['r']
h_traces = traces['h']


A_mean = np.mean(A_traces,0)
B_mean = np.mean(B_traces,0)
C_mean = np.mean(C_traces,0)
D_mean = np.mean(D_traces,0)
h_mean = np.mean(h_traces,0)
r_mean = np.mean(r_traces,0)
LQ_mean = np.mean(LQ_traces,0)

h_upper_ci = np.percentile(h_traces, 97.5, axis=0)
h_lower_ci = np.percentile(h_traces, 2.5, axis=0)



plt.subplot(1,1,1)
plt.plot(y_est,linewidth=0.5)
plt.plot(yhat_mean,linewidth=0.5)
plt.plot(yhat_upper_ci,'--',linewidth=0.5)
plt.plot(yhat_lower_ci,'--',linewidth=0.5)
plt.title('measurement estimates')
plt.legend(('true','mean','upper CI','lower CI'))
plt.show()



def plot_trace(param,num_plots,pos, param_name='parameter'):
    """Plot the trace and posterior of a parameter."""

    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

    # Plotting
    plt.subplot(num_plots, 1, pos)
    plt.hist(param, 30, density=True)
    sns.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.axvline(mean, color='r', lw=2, linestyle='--', label='mean')
    plt.axvline(median, color='c', lw=2, linestyle='--', label='median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()

plot_trace(A_traces[:,1,0],4,1,'A[2,2]')
plot_trace(C_traces[:,1],4,2,'C[1]')
plot_trace(D_traces,4,3,'D')
plot_trace(r_traces,4,4,'r')
# # plot_trace(q_traces,5,5,'q')
plt.show()
#
plt.subplot(1,1,1)
plt.plot(A_traces[:,1,0],A_traces[:,1,1],'o')
plt.title('samples pairs plot')
plt.show()


# how to do actual validation?? BODE diagram
# B_mean = np.array([[0],[1]])
A_true = data['A_true']
B_true = data['B_true']
C_true = data['C_true']
D_true = data['D_true']

w_plot = np.logspace(-2,1)

# plot estimated bode diagram samples
no_samples = np.shape(A_traces)[0]
no_plot = 100
sel = np.random.choice(np.arange(no_samples),no_plot,False)

for s in sel:
    A_sample = A_traces[s,:,:]
    B_sample = B_traces[s,:].reshape(2,1)
    C_sample = C_traces[s,:].reshape(1,2)
    D_sample = D_traces[s]
    w, mag, phase = signal.bode((A_sample, B_sample, C_sample, float(D_sample)),w_plot)

    plt.subplot(2,1,1)
    h2, = plt.semilogx(w, mag,color='green',alpha=0.1,label='samples')    # Bode magnitude plot
    plt.subplot(2,1,2)
    plt.semilogx(w, phase,color='green',alpha=0.1)  # Bode phase plot


# plot the true bode diagram
w,mag,phase = signal.bode((A_true,B_true,C_true,float(D_true)),w_plot)
# have to convert to flaot because for some reason D, and C are uint8
plt.subplot(2,1,1)
h1, = plt.semilogx(w, mag,color='blue', label='True system')    # Bode magnitude plot
plt.title('Bode diagram')
plt.ylabel('Magnitude (dB)')
plt.legend(handles=[h1,h2])
plt.subplot(2,1,2)
plt.semilogx(w, phase,color='blue')  # Bode phase plot
plt.ylabel('Phase (deg)')
plt.xlabel('Frequency (rad/s)')



plt.show()



