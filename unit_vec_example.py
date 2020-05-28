import pystan
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from helpers import plot_trace

np.random.seed(12)

T = 100
r = 0.2
theta = np.linspace(np.pi,3*np.pi,T)

y = (theta + np.random.normal(0.0, r,np.shape(theta)) ) % (2 * np.pi) - np.pi

theta = theta % (2 * np.pi) -np.pi

model = pystan.StanModel(file='stan/unit_vec_ex.stan')

stan_data = {
    'N':T,
    'y':y,
}

fit = model.sampling(data=stan_data)

traces = fit.extract()

theta_hat = traces['theta']
omega_hat = traces['omega']
theta_hat_mean = np.mean(theta_hat, axis=0)

plt.plot(theta)
plt.plot(y)
plt.plot(theta_hat_mean)
plt.show()

plt.plot(np.mean(omega_hat,axis=0))
plt.show()