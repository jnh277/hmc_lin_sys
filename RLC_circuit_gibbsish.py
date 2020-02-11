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

"""Estimates a RLC circuit and its states."""
import pystan
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path


# specific data path
data_path = 'data/rlc_circuit.mat'
data = loadmat(data_path)

y_est = data['y_estimation'].flatten()
u_est = data['u_estimation'].flatten()
y_val = data['y_validation'].flatten()
u_val = data['u_validation'].flatten()
Ts = data['Ts'].flatten()
states = data['states_est']

no_obs_est = len(y_est)
no_obs_val = len(y_val)


# this starts all the parameters close to the true values
# Cq_init = (data["Cq"] * np.random.uniform(0.8, 1.2, 1))[0,0]
# Lq_init = (data["Lq"] * np.random.uniform(0.8, 1.2, 1))[0,0]
# Rq_init = (data["Rq"] * np.random.uniform(0.8, 1.2, 1))[0,0]
# q_init = (np.sqrt(data["Q"]) * np.random.uniform(0.8, 1.2, 1))[0,0]
# r_init = (np.sqrt(data["R"]) * np.random.uniform(0.8, 1.2, 1))[0,0]

# starts all parameters a bit further away
Cq_init = (data["Cq"] * np.random.uniform(0.5, 1.5, 1))[0,0]
Lq_init = (data["Lq"] * np.random.uniform(0.5, 1.5, 1))[0,0]
Rq_init = (data["Rq"] * np.random.uniform(0.5, 1.5, 1))[0,0]
q_init = (np.sqrt(data["Q"]) * np.random.uniform(0.5, 1.5, 1))[0,0]
r_init = (np.sqrt(data["R"]) * np.random.uniform(0.5, 1.5, 1))[0,0]

# this starts sys parameters close but noise parameters further away
# Cq_init = (data["Cq"] * np.random.uniform(0.8, 1.2, 1))[0,0]
# Lq_init = (data["Lq"] * np.random.uniform(0.8, 1.2, 1))[0,0]
# Rq_init = (data["Rq"] * np.random.uniform(0.8, 1.2, 1))[0,0]
# q_init = 1.0
# r_init = 1.0


# load models
model1 = pickle.load(open('stan/RLC_circuit_filter.pkl', 'rb'))
model2 = pickle.load(open('stan/RLC_circuit_paramID.pkl', 'rb'))

# inital sampling of hidden states

def stan_data_f1(Cq,Rq,Lq,q,r):
    stan_data = {'no_obs_est': len(y_est),
                 'no_obs_val': len(y_val),
                 'y_est': y_est,
                 'u_est': u_est,
                 'y_val': y_val,
                 'u_val': u_val,
                 'Ts': Ts[0],
                 'Cq': Cq_init,
                 'Rq': Rq_init,
                 'Lq': Lq_init,
                 'q': q_init,
                 'r': r_init,
                 }
    return stan_data

stan_data = stan_data_f1(Cq_init,Rq_init,Lq_init,q_init,r_init)
fit = model1.sampling(data=stan_data, iter=510, chains=1,warmup=500)
traces = fit.extract()
h_traces = traces["h"]
h_traces_mean = np.mean(h_traces,0)

# initial sampling of parameters
def stan_data_f2(states):
    stan_data = {'no_obs_est': len(y_est),
             'no_obs_val': len(y_val),
             'y_est': y_est,
             'u_est':u_est,
             'y_val':y_val,
             'u_val':u_val,
             'Ts':Ts[0],
             'h':states,
             }
    return stan_data

stan_data = stan_data_f2(h_traces[-1,:,:])

def init_function2():
    output = dict(r=r_init,
                  q=q_init,
                  Cq=Cq_init,
                  Rq=Rq_init,
                  Lq = Lq_init,
                  )
    return output

fit = model2.sampling(data=stan_data,init=init_function2, iter=510, chains=1,warmup=500)
traces = fit.extract()
Cq_traces = traces['Cq']
Rq_traces = traces['Rq']
Cq_traces = traces['Cq']
Rq_traces = traces['Rq']
Lq_traces = traces['Lq']
q_traces = traces['q']
r_traces = traces['r']

iters = 1000

Cq_samples = np.zeros((iters,1))
Rq_samples = np.zeros((iters,1))
Lq_samples = np.zeros((iters,1))
q_samples = np.zeros((iters,1))
r_samples = np.zeros((iters,1))
h_samples = np.zeros((iters,2,no_obs_est))

h_samples[0,:,:] = h_traces[-1,:,:]
Cq_samples[0] = Cq_traces[-1]
Rq_samples[0] = Rq_traces[-1]
Lq_samples[0] = Lq_traces[-1]
q_samples[0] = q_traces[-1]
r_samples[0] = r_traces[-1]

for iter in range(1,iters):         # first iter already done
    print('Running iter ', iter)
    # sample states
    def init_function1():
        output = dict(h=h_samples[iter-1,:,:],
                      )
        return output
    stan_data = stan_data_f1(Cq_samples[iter-1], Rq_samples[iter-1], Lq_samples[iter-1], q_samples[iter-1], r_samples[iter-1])
    fit = model1.sampling(data=stan_data, init=init_function1, iter=250, chains=1, warmup=150, verbose=False)
    traces = fit.extract()
    h_traces = traces["h"]
    h_samples[iter,:,:] = h_traces[-1,:,:]

    # sample parameters
    stan_data = stan_data_f2(h_samples[iter-1, :, :])
    def init_function2():
        output = dict(r=r_samples[iter-1][0],
                      q=q_samples[iter-1][0],
                      Cq=Cq_samples[iter-1][0],
                      Rq=Rq_samples[iter-1][0],
                      Lq=Lq_samples[iter-1][0],
                      )
        return output
    fit = model2.sampling(data=stan_data, init=init_function2, iter=250, chains=1, warmup=150, verbose=False)
    traces = fit.extract()
    Cq_traces = traces['Cq']
    Rq_traces = traces['Rq']
    Cq_traces = traces['Cq']
    Rq_traces = traces['Rq']
    Lq_traces = traces['Lq']
    q_traces = traces['q']
    r_traces = traces['r']
    Cq_samples[iter] = Cq_traces[-1]
    Rq_samples[iter] = Rq_traces[-1]
    Lq_samples[iter] = Lq_traces[-1]
    q_samples[iter] = q_traces[-1]
    r_samples[iter] = r_traces[-1]

# print(fit)

# yhat = traces['y_hat']
# yhat[np.isnan(yhat)] = 0.0
# yhat[np.isinf(yhat)] = 0.0
#
# yhat_mean = np.mean(yhat, axis=0)
# yhat_upper_ci = np.percentile(yhat, 97.5, axis=0)
# yhat_lower_ci = np.percentile(yhat, 2.5, axis=0)



# h_traces = traces['h']
#
#
# Cq_mean = np.mean(Cq_traces,0)
# Rq_mean = np.mean(Rq_traces,0)
# Lq_mean = np.mean(Lq_traces,0)
# h_mean = np.mean(h_traces,0)
# r_mean = np.mean(r_traces,0)
# q_mean = np.mean(q_traces,0)

#
#
# plt.subplot(1,1,1)
# plt.plot(y_est,linewidth=0.5)
# plt.plot(yhat_mean,linewidth=0.5)
# plt.plot(yhat_upper_ci,'--',linewidth=0.5)
# plt.plot(yhat_lower_ci,'--',linewidth=0.5)
# plt.show()
#
#
#
h1 = h_samples[:,0,:]
#
plt.subplot(1,1,1)
plt.plot(states[0,:],linewidth=0.5)
plt.plot(np.mean(h_samples[:,0,:],0),linewidth=0.5)
plt.plot(np.percentile(h1, 97.5, axis=0),'--',linewidth=0.5)
plt.plot(np.percentile(h1, 2.5, axis=0),'--',linewidth=0.5)
plt.title('state 1 estimation')
plt.show()
#
h2 = h_samples[:,1,:]

plt.subplot(1,1,1)
plt.plot(states[1,:],linewidth=0.5)
plt.plot(np.mean(h2,0),linewidth=0.5)
plt.plot(np.percentile(h2, 97.5, axis=0),'--',linewidth=0.5)
plt.plot(np.percentile(h2, 2.5, axis=0),'--',linewidth=0.5)
plt.title('state 2 estimation')
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
    # sns.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.axvline(mean, color='r', lw=2, linestyle='--', label='mean')
    plt.axvline(median, color='c', lw=2, linestyle='--', label='median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()

plot_trace(Cq_samples,4,1,'Cq')
plot_trace(Rq_samples,4,2,'Rq')
plot_trace(Lq_samples,4,3,'Lq')
plot_trace(r_samples,4,4,'r')
# plot_trace(q_traces,5,5,'q')
plt.show()

plt.subplot(1,1,1)
plt.plot(Cq_samples,Rq_samples,'o')
plt.xlabel('Cq')
plt.ylabel('Rq')
plt.title('sample pairs plot')
plt.show()


