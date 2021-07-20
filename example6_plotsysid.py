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

""" Plots the model estimated by example6_sysid.py, these plots correspond to the
    system identification (bode response) results shown in example 6 of the paper
     (Section 6.7).
     This also converts the estimated samples from state space form to transfer function form
     so that the mean transfer function can be calculated and used for control design in the
     matlab script matlab/example6_controldesign.m"""


import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt
from helpers import plot_trace
import pickle
from scipy.signal import ss2tf
from scipy import signal


# specific data path
data_path = 'data/control_example_data.mat'
data = loadmat(data_path)

y_est = data['y_estimation'].flatten()
u_est = data['u_estimation'].flatten()
# states_est = data['states_est']
Ts = data['Ts'].flatten()

no_obs_est = len(y_est)

with open('results/ctrl_sysid_traces2.pickle', 'rb') as file:
    traces = pickle.load(file)

yhat = traces['y_hat'].swapaxes(0,-1)
yhat[np.isnan(yhat)] = 0.0
yhat[np.isinf(yhat)] = 0.0

yhat_mean = np.mean(yhat, axis=0)
yhat_upper_ci = np.percentile(yhat, 97.5, axis=0)
yhat_lower_ci = np.percentile(yhat, 2.5, axis=0)


A_traces = traces['A'].swapaxes(0,-1)
B_traces = traces['B'].swapaxes(0,-1)
C_traces = traces['C'].swapaxes(0,-1)
D_traces = traces['D'].swapaxes(0,-1)
LQ_traces = traces['LQ'].swapaxes(0,-1)
r_traces = traces['r'].swapaxes(0,-1)
h_traces = traces['h'].swapaxes(0,-1)


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




plot_trace(A_traces[:,1,0],4,1,'A[2,2]')
plot_trace(C_traces[:,1],4,2,'C[1]')
plot_trace(D_traces,4,3,'D')
plot_trace(r_traces,4,4,'r')
plt.show()
#
# plt.subplot(1,1,1)
# plt.plot(A_traces[:,1,0],A_traces[:,1,1],'o')
# plt.title('samples pairs plot')
# plt.show()


# BODE diagram
# B_mean = np.array([[0],[1]])
A_true = data['a']
B_true = data['b']
C_true = data['c']
D_true = data['d']

w_plot = np.logspace(-3,0.5)
#
A_ML = data['A_ML']
B_ML = data['B_ML']
C_ML = data['C_ML']
D_ML = data['D_ML']



def plot_bode_ML(A_smps, B_smps, C_smps, D_smps, A_t, B_t, C_t, D_t, A_ML, B_ML, C_ML, D_ML, omega, no_plot=300, max_samples=1000, save=False):
    """plot bode diagram from estimated system samples and true sys and maximum likelihood estiamte"""
    no_samples = np.shape(A_smps)[0]
    n_states = np.shape(A_smps)[1]
    no_eval = max(no_samples, max_samples)
    sel = np.random.choice(np.arange(no_samples), no_eval, False)
    omega_res = max(np.shape(omega))

    mag_samples = np.zeros((omega_res, no_eval))
    phase_samples = np.zeros((omega_res, no_eval))

    count = 0
    for s in sel:
        A_s = A_smps[s]
        B_s = B_smps[s].reshape(n_states, -1)
        C_s = C_smps[s].reshape(-1, n_states)
        D_s = D_smps[s]
        w, mag_samples[:, count], phase_samples[:, count] = signal.bode((A_s, B_s, C_s, float(D_s)), omega)
        count = count + 1

    # now what if i also want to show the MAP estimate
    # no_freqs = np.shape(mag_samples)[0]
    # mag_MAP = np.zeros((no_freqs))
    # phase_MAP = np.zeros((no_freqs))
    # for k in range(no_freqs):
    #     mag_MAP[k] = calc_MAP(mag_samples[k, :])
    #     phase_MAP[k] = calc_MAP(phase_samples[k, :])

    w, mag_true, phase_true = signal.bode((A_t, B_t, C_t, float(D_t)), omega)
    w, mag_ML, phase_ML = signal.bode((A_ML, B_ML, C_ML, float(D_ML)), omega)

    # plot the samples
    plt.subplot(2, 1, 1)
    h2, = plt.semilogx(w.flatten(), mag_samples[:, 0], color='green', alpha=0.1,
                       label='posterior samples')  # Bode magnitude plot
    plt.semilogx(w.flatten(), mag_samples[:, 1:no_plot], color='green', alpha=0.1)  # Bode magnitude plot
    h1, = plt.semilogx(w.flatten(), mag_true, color='black', label='True system')  # Bode magnitude plot
    hml, = plt.semilogx(w.flatten(), mag_ML,'--', color='purple', label='ML estimate')  # Bode magnitude plot
    hm, = plt.semilogx(w.flatten(), np.mean(mag_samples, 1), '-.', color='orange',label='conditional mean')  # Bode magnitude plot
    # hmap = plt.semilogx(w.flatten(), mag_MAP, '-.', color='blue',label='hmc MAP')  # Bode magnitude plot

    # hu, = plt.semilogx(w.flatten(), np.percentile(mag_samples, 97.5, axis=1),'--',color='orange',label='Upper CI')    # Bode magnitude plot

    # plt.legend(handles=[h1, h2, hml, hm, hmap])
    plt.legend()
    plt.title('Bode diagram')
    plt.ylabel('Magnitude (dB)')
    plt.xlim((min(omega), max(omega)))

    plt.subplot(2, 1, 2)
    plt.semilogx(w.flatten(), phase_samples[:, :no_plot], color='green', alpha=0.1)  # Bode phase plot
    plt.semilogx(w.flatten(), phase_true, color='black')  # Bode phase plot
    plt.semilogx(w.flatten(), phase_ML,'--', color='purple')  # Bode phase plot
    plt.semilogx(w.flatten(), np.mean(phase_samples, 1), '-.', color='orange',
                 label='mean')  # Bode magnitude plot
    # plt.semilogx(w.flatten(), phase_MAP, '-.', color='blue',
    #              label='map')  # Bode magnitude plot
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/s)')
    plt.xlim((min(omega), max(omega)))

    if save:
        plt.savefig('figures/ctrl_example_bode.png',format='png')

    plt.show()


plot_bode_ML(A_traces,B_traces,C_traces,D_traces,A_true,B_true,C_true,D_true,A_ML,B_ML,C_ML,D_ML,w_plot,save=True)

# convert to transfer function
num_samples = np.shape(A_traces)[0]


# for i in range(num_samples):
tf_nums = np.zeros((num_samples,7))
tf_dens = np.zeros((num_samples,7))
for i in range(num_samples):
    tf = ss2tf(A_traces[i,:,:],np.expand_dims(B_traces[i,:],1),np.expand_dims(C_traces[i,:],0),float(D_traces[i]))
    tf_nums[i,:] = tf[0]
    tf_dens[i,:] = tf[1]

tf_num_mean = np.mean(tf_nums,0)
tf_den_mean = np.mean(tf_dens,0)

w, mag_mean, phase_mean = signal.bode((tf_num_mean,tf_den_mean))

#
# plt.semilogx(w,mag_mean)
# plt.show()

hmc_sysid_results = {"A_traces":A_traces,
                     "B_traces":B_traces,
                     "C_traces":C_traces,
                     "D_traces":D_traces,
                     "r_traces":r_traces,
                     "LQ_traces":LQ_traces,
                     "tf_nums":tf_nums,
                     "tf_dens":tf_dens,
                     "tf_num_mean":tf_num_mean,
                     "tf_den_mean":tf_den_mean,
                     }
savemat('results/ctrl_example_sysid.mat',hmc_sysid_results)