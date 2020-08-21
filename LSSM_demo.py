import pystan
import numpy as np
import matplotlib.pyplot as plt
from helpers import plot_trace

def ssm(x, a=0.9):
    return a*x

N = 300         # number of time steps
x0 = 3.0        # initial x
r = 0.1         # measurement noise standard deviation
q = 0.2         # process noise standard deviation



# simulate the system
x = np.ones(N)
x[0] = x0
for k in range(N-1):
    x[k+1] = ssm(x[k]) + np.random.normal(0.0, q)

# simulate measurements
y = x + np.random.normal(0.0, r,np.shape(x))

# model = pystan.StanModel(file='stan/LSSM_demo.stan')
model = pystan.StanModel(file='stan/LSSM_error_param.stan')

stan_data = {
    'N':N,
    'y':y,
}

fit = model.sampling(data=stan_data)

traces = fit.extract()

z = traces['z']
z_mean = np.mean(z,0)
# z_upper = np.percentile(z,)

a = traces['a']
a_mean = np.mean(a,0)
r = traces['r']
q = traces['q']

plot_trace(a,3,1,'a')
plot_trace(r,3,2,'r')
plot_trace(q,3,3,'q')
plt.show()

plt.subplot(1,1,1)
plt.plot(x)
plt.plot(z_mean,'--')
plt.plot(y,'o')
plt.show()


