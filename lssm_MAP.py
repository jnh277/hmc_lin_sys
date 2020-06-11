import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
# from helpers import plot_trace
# from helpers import plot_bode_ML
# from lssm import run_lssm_hmc
import pickle
import seaborn as sns
from scipy import signal
from scipy.stats import gaussian_kde as kde


# specific data path
data_path = 'data/example4_lssm.mat'
data = loadmat(data_path)

y_est = data['y_estimation'].flatten()
u_est = data['u_estimation'].flatten()
states_est = data['states_est']
Ts = data['Ts'].flatten()

no_obs_est = len(y_est)



with open('results/lssm_traces.pickle', 'rb') as file:
    traces = pickle.load(file)


A_traces = traces['A']
B_traces = traces['B']
C_traces = traces['C']
D_traces = traces['D']
LQ_traces = traces['LQ']
r_traces = traces['r']
h_traces = traces['h']


A_true = data['A_true']
B_true = data['B_true']
C_true = data['C_true']
D_true = data['D_true']

w_plot = np.logspace(-2,3)
#
A_ML = data['A_ML']
B_ML = data['B_ML']
C_ML = data['C_ML']
D_ML = data['D_ML']


def calc_MAP(x):
    min_x = np.min(x)
    max_x = np.max(x)
    pos = np.linspace(min_x, max_x, 100)
    kernel = kde(x)
    z = kernel(pos)
    return pos[np.argmax(z)]



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
    no_freqs = np.shape(mag_samples)[0]
    mag_MAP = np.zeros((no_freqs))
    phase_MAP = np.zeros((no_freqs))
    for k in range(no_freqs):
        mag_MAP[k] = calc_MAP(mag_samples[k, :])
        phase_MAP[k] = calc_MAP(phase_samples[k, :])



    w, mag_true, phase_true = signal.bode((A_t, B_t, C_t, float(D_t)), omega)
    w, mag_ML, phase_ML = signal.bode((A_ML, B_ML, C_ML, float(D_ML)), omega)

    # plot the samples
    plt.subplot(2, 1, 1)
    h2, = plt.semilogx(w.flatten(), mag_samples[:, 0], color='green', alpha=0.1,
                       label='hmc samples')  # Bode magnitude plot
    plt.semilogx(w.flatten(), mag_samples[:, 1:no_plot], color='green', alpha=0.1)  # Bode magnitude plot
    h1, = plt.semilogx(w.flatten(), mag_true, color='black', label='True system')  # Bode magnitude plot
    hml, = plt.semilogx(w.flatten(), mag_ML,'--', color='purple', label='ML estimate')  # Bode magnitude plot
    hm, = plt.semilogx(w.flatten(), np.mean(mag_samples, 1), '-.', color='orange',label='hmc mean')  # Bode magnitude plot
    hmap = plt.semilogx(w.flatten(), mag_MAP, '-.', color='blue',label='hmc MAP')  # Bode magnitude plot

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
    plt.semilogx(w.flatten(), phase_MAP, '-.', color='blue',
                 label='map')  # Bode magnitude plot
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/s)')
    plt.xlim((min(omega), max(omega)))

    if save:
        plt.savefig('figures/example4_bode.png',format='png')

    plt.show()

    return mag_samples, phase_samples, w

mag_samples, phase_samples, w = plot_bode_ML(A_traces,B_traces,C_traces,D_traces,A_true,B_true,C_true,D_true,A_ML,B_ML,C_ML,D_ML,w_plot,save=False)

plt.hist(mag_samples[:])