import pystan
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from helpers import plot_trace


xp = np.linspace(0,2*3.14,100)
f = np.sin(xp)

N = 50
sig_e = 0.1

nu = 1.0

# x = np.random.uniform(0,2*3.14,(N))
x = np.linspace(0,2*3.14,N)
# y = np.sin(x) + np.random.normal(0,sig_e,(N))
y = np.sin(x) + sig_e*np.random.standard_t(nu,(N))

Np = len(xp)

model = pystan.StanModel(file='stan/gpT.stan')


# stan_data = {'N':N,
#              'x':x,
#              'y':y
#              }

stan_data = {'N':N,
             'x':x,
             'y':y,
             'nu':1.0,
             }

fit = model.sampling(data=stan_data, iter=2000, chains=2)


traces = fit.extract()

rho = traces['rho']
sigma = traces['sigma']
alpha = traces['alpha']
fhat = traces['f']
fhat_mean = np.mean(fhat,0)

plt.subplot(1,1,1)
plt.plot(xp,f)
plt.plot(x,y,'o')
plt.plot(x,fhat_mean,'--')
plt.ylim((-1.4,1.4))
plt.title('GP')
plt.show()

plot_trace(rho,3,1,'rho')
plt.title('GP')
plot_trace(sigma,3,2,'sigma')
plot_trace(alpha,3,3,'alpha')

plt.show()

