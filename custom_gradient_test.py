import stan
import numpy as np
import matplotlib.pyplot as plt

N = 50
theta = 2
x = np.linspace(0, 1.0, N)
y = np.power(theta * x, 2) + np.random.normal(0, 0.1, (N,))

plt.plot(x, y)
plt.show()

model_code = """
functions {
    real myfunc(real x);
}

data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}

parameters {
    real <lower=0> theta;
    real <lower=1e-8> r;
}
transformed parameters {
    vector[N] fhat = (theta * x) .* (theta * x);
}

model{
    r ~ cauchy(0.0, 1.0);
    y ~ normal(fhat, r);
}

"""

stan_data = {
    'N':N,
    'x':x,
    'y':y
}

posterior = stan.build(model_code, stanc_options=list("allow_undefined"), includes=["custom_grad_test.hpp"], include_dirs=["."], data=stan_data, random_seed=1120)
fit = posterior.sample(num_samples=2000)

theta_samp = fit['theta'][0,:]

plt.hist(theta_samp, bins=30)
plt.show()
