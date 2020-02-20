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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

def build_phi_matrix(obs,order,inputs):
    "Builds the regressor matrix"
    no_obs = len(obs)
    max_delay = np.max((order[0],order[1]-1))
    phi = np.zeros((no_obs-max_delay, np.sum(order)))
    for i in range(order[0]):
        phi[:,i] = obs[max_delay-i-1:-i-1]
    for i in range(order[1]):
        phi[:,i+order[0]] = inputs[max_delay-i:no_obs-i]
    return phi

def build_input_matrix(inputs, input_order):
    no_obs = len(inputs)
    max_delay = input_order - 1
    phi = np.zeros((no_obs-max_delay, input_order))
    for i in range(input_order):
        phi[:, i] = inputs[max_delay - i:no_obs - i]
    return phi

def build_obs_matrix(obs, output_order):
    no_obs = len(obs)
    max_delay = output_order
    phi = np.zeros((no_obs - max_delay, output_order))
    for i in range(output_order):
        phi[:,i] = obs[max_delay-i-1:-i-1]
    return phi


def generate_data(no_obs, a, b, sigmae):
    order_a = len(a)
    order_b = len(b)
    order_ab = order_a + order_b
    order_max = np.max((order_a, order_b))
    y = np.zeros(no_obs)
    u = np.random.normal(size=no_obs)

    y[0] = 0.0
    Phi = np.zeros((no_obs - order_max, order_ab))

    for t in range(order_max, no_obs):
        y[t] = np.sum(a * y[range(t-1, t-order_a, -1)])
        y[t] += np.sum(b * u[range(t-1, t-order_b, -1)])
        y[t] += sigmae * np.random.normal()
        Phi[t - order_max, :] = np.hstack((y[range(t-1, t-order_a-1, -1)], u[range(t-1, t-order_b-1, -1)]))

    return y, u, Phi


def plot_trace(param,num_plots,pos, param_name='parameter'):
    """Plot the trace and posterior of a parameter."""

    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

    # Plotting
    plt.subplot(num_plots, 1, pos)
    plt.hist(param, 30, density=True);
    sns.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.axvline(mean, color='r', lw=2, linestyle='--', label='mean')
    plt.axvline(median, color='c', lw=2, linestyle='--', label='median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()


def plot_dbode(num_samples,den_samples,num_true,den_true,Ts,omega,no_plot=300):
    # plot estimated bode diagram samples
    no_samples = np.shape(num_samples)[0]
    sel = np.random.choice(np.arange(no_samples), no_plot, False)
    omega_res = max(np.shape(omega))

    mag_samples = np.zeros((omega_res, no_plot))
    phase_samples = np.zeros((omega_res, no_plot))


    count = 0
    for s in sel:
        num_sample = np.concatenate(([1.0], num_samples[s,:]), 0)
        den_sample = den_samples[s, :]  # need to flip vectors
        w, mag_samples[:, count], phase_samples[:, count] = signal.dbode((num_sample, den_sample, Ts), omega)
        count = count + 1

    # calculate the true bode diagram
    # plot the true bode diagram
    w, mag_true, phase_true = signal.dbode((num_true.flatten(), den_true.flatten(), Ts), omega)

    # plot the samples
    plt.subplot(2, 1, 1)
    h2, = plt.semilogx(w.flatten(), mag_samples[:, 0], color='green', alpha=0.1, label='samples')  # Bode magnitude plot
    plt.semilogx(w.flatten(), mag_samples[:, 1:], color='green', alpha=0.1)  # Bode magnitude plot
    h1, = plt.semilogx(w.flatten(), mag_true, color='blue', label='True system')  # Bode magnitude plot
    hm, = plt.semilogx(w.flatten(), np.mean(mag_samples, 1), '-.', color='orange', label='mean')  # Bode magnitude plot
    # hu, = hm, = plt.semilogx(w.flatten(), np.percentile(mag_samples, 97.5, axis=1),'--',color='orange',label='Upper CI')    # Bode magnitude plot

    plt.legend(handles=[h1, h2, hm])
    plt.legend()
    plt.title('Bode diagram')
    plt.ylabel('Magnitude (dB)')
    plt.xlim((10e-2, 10e1))

    plt.subplot(2, 1, 2)
    plt.semilogx(w.flatten(), phase_samples, color='green', alpha=0.1)  # Bode phase plot
    plt.semilogx(w.flatten(), phase_true, color='blue')  # Bode phase plot
    plt.semilogx(w.flatten(), np.mean(phase_samples, 1), color='green', alpha=0.1)  # Bode phase plot
    hm, = plt.semilogx(w.flatten(), np.mean(phase_samples, 1), '-.', color='orange',
                       label='mean')  # Bode magnitude plot
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/s)')
    plt.xlim((10e-2, 10e1))
    plt.ylim((-330, 60))

    # plt.show()

