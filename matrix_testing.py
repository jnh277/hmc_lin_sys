import pystan
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns


no_obs_est = 200
dims = 2

Sigma = np.array([[1, -0.25],[-0.25,1]])
A = np.array([[2.0, 0.5],[0.5, 1.0]])
mean = np.array([0, 0])

u = np.random.multivariate_normal(mean, Sigma, no_obs_est)
mu = np.matmul(A,u.T).T
y = np.zeros((no_obs_est, dims))
for i in range(no_obs_est):
    y[i,:] = np.random.multivariate_normal(mu[i,:],Sigma)


def init_function():
    output = dict(L_A=np.array([[1.0, 0.0],[0.0,1.0]]),
                  )
    return output

model = pystan.StanModel(file='stan/wishart_test.stan')

stan_data = {'no_obs': int(no_obs_est),
             'obs':y.T,
             'K':int(dims),
             'u':u.T
             }

fit = model.sampling(data=stan_data,init=init_function, iter=5000, chains=4)
traces = fit.extract()

A_hat = np.mean(traces["A"],0)
Sigma_hat = np.mean(traces["Sigma"],0)