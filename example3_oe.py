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

"""Estimates an OE model using data with Gaussian noise and horeshoe priors."""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from helpers import plot_trace
from scipy import signal
from oe import run_oe_hmc


# specific data path
data_path = 'data/example3_oe.mat'
# input_order = 4
# output_order = 3
input_order = 11
output_order = 10

data = loadmat(data_path)
y_val = data['y_validation'].flatten()

(fit, traces) = run_oe_hmc(data_path, input_order, output_order, iter=10000)

yhat = traces['y_hat_val']
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0

# yhat_OL = traces['y_hat_val2']
# yhat_OL_mean = np.mean(yhat_OL, axis=0)

yhat_mean = np.mean(yhat, axis=0)
yhat_upper_ci = np.percentile(yhat, 97.5, axis=0)
yhat_lower_ci = np.percentile(yhat, 2.5, axis=0)

MF_hmc = 100*(1-np.sum(np.power(y_val[10:]-yhat_mean[10:],2))/np.sum(np.power(y_val[10:],2)))


# mu_traces = traces['mu']
f_coef_traces = traces['f_coefs']
b_coef_traces = traces['b_coefs']
# shrinkage_param = traces["shrinkage_param"]
# shrinkage_param_mean = np.mean(shrinkage_param,0)
# r_traces = traces["r"]

# mu_mean = np.mean(mu_traces,0)
# r_mean = np.mean(r_traces,0)

f_mean = np.mean(f_coef_traces,0)
b_mean = np.mean(b_coef_traces,0)

plt.subplot(1,1,1)
plt.plot(y_val,linewidth=0.5)
plt.plot(yhat_mean,linewidth=0.5)
# plt.plot(yhat_OL_mean,linewidth=0.5)
# plt.plot(yhat_upper_ci,'--',linewidth=0.5)
# plt.plot(yhat_lower_ci,'--',linewidth=0.5)
plt.ylim((-2,2))
plt.legend(('y val','y hat'))
plt.show()



plot_trace(f_coef_traces[:,0],4,1,'f[0]')
plot_trace(f_coef_traces[:,1],4,2,'f[2]')
plot_trace(b_coef_traces[:,0],4,3,'b[0]')
plot_trace(b_coef_traces[:,1],4,4,'b[1]')
plt.show()


# now plot the bode diagram

f_true = data["f_true"]
b_true = data["b_true"]

Ts = 1
w_res = 100
w_plot = np.logspace(-3,np.log10(10*3.14),w_res)


F_ML = data['f_ml']
B_ML = data['b_ml']



def plot_dbode_ML(num_samples,den_samples,num_true,den_true,num_ML,den_ML,Ts,omega,no_plot=300, max_samples=1000, save=False):
    """plot bode diagram from estimated discrete time system samples and true sys"""
    no_samples = np.shape(num_samples)[0]
    no_eval = min(no_samples,max_samples)
    sel = np.random.choice(np.arange(no_samples), no_eval, False)
    omega_res = max(np.shape(omega))

    mag_samples = np.zeros((omega_res, no_eval))
    phase_samples = np.zeros((omega_res, no_eval))



    count = 0
    for s in sel:
        den_sample = np.concatenate(([1.0], den_samples[s,:]), 0)
        num_sample = num_samples[s, :]
        w, mag_samples[:, count], phase_samples[:, count] = signal.dbode((num_sample, den_sample, Ts), omega)
        count = count + 1

    # calculate the true bode diagram
    # plot the true bode diagram
    w, mag_true, phase_true = signal.dbode((num_true.flatten(), den_true.flatten(), Ts), omega)
    w, mag_ML, phase_ML = signal.dbode((num_ML.flatten(), den_ML.flatten(), Ts), omega)

    # convert back from decibal
    mag_true = np.power(10,mag_true/10)
    mag_ML = np.power(10, mag_ML / 10)
    mag_samples = np.power(10, mag_samples / 10)

    # plot the samples
    plt.subplot(2, 1, 1)
    h2, = plt.semilogx(w.flatten(), mag_samples[:, 0], color='green', alpha=0.1, label='hmc samples')  # Bode magnitude plot
    plt.semilogx(w.flatten(), mag_samples[:, 1:no_plot], color='green', alpha=0.1)  # Bode magnitude plot
    h1, = plt.semilogx(w.flatten(), mag_true, color='blue', label='True system')  # Bode magnitude plot
    hml, = plt.semilogx(w.flatten(), mag_ML,'--', color='purple', label='ML estimate')  # Bode magnitude plot
    hm, = plt.semilogx(w.flatten(), np.mean(mag_samples, 1), '-.', color='orange', label='hmc mean')  # Bode magnitude plot
    # hu, = plt.semilogx(w.flatten(), np.percentile(mag_samples, 97.5, axis=1),'--',color='orange',label='Upper CI')    # Bode magnitude plot

    plt.legend(handles=[h1, h2, hm])
    plt.legend()
    plt.title('Bode diagram')
    plt.ylabel('Magnitude (dB)')
    plt.xlim((min(w.flatten()),min(max(omega),1/Ts*3.14)))

    plt.subplot(2, 1, 2)
    plt.semilogx(w.flatten(), phase_samples[:,:no_plot], color='green', alpha=0.1)  # Bode phase plot
    plt.semilogx(w.flatten(), phase_true, color='blue')  # Bode phase plot
    hml, = plt.semilogx(w.flatten(), phase_ML, '--', color='purple')  # Bode magnitude plot
    plt.semilogx(w.flatten(), np.mean(phase_samples, 1), '-.', color='orange',
                       label='mean')  # Bode magnitude plot
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/s)')
    plt.xlim((min(w.flatten()), min(max(omega),1/Ts*3.14)))
    plt.ylim((-500, 50))

    if save:
        plt.savefig('bode_plot_oe.png',format='png')

    plt.show()


plot_dbode_ML(b_coef_traces[:,-1::-1],f_coef_traces[:,-1::-1],b_true.flatten(),f_true.flatten(),B_ML.flatten(),F_ML.flatten(),Ts,w_plot,save=False)



