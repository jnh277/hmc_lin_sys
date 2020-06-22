import numpy as np
import pystan
import matplotlib.pyplot as plt

# define some things
N = 20
a = 0.9
sig_e = 0.2
y0 = 1

# generate some data
y = np.zeros((N))
y[0] = y0

for i in range(N-1):
    y[i+1] = a*y[i] + np.random.normal(0, sig_e, 1)


plt.plot(y)
plt.show()

stan_data = {
    'y':y,
    'N':N,
    'sig_e':sig_e
}

model = pystan.StanModel(file='./stan/ar_demo.stan')

fit = model.sampling(data=stan_data)

traces = fit.extract()

a_samples = traces['a']

plt.subplot(2,1,1)
plt.plot(a_samples)
plt.subplot(2,1,2)
plt.hist(a_samples)
