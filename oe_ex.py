###############################################################################
#    Practical Bayesian System identification using Hamiltonian Monte Carlo
#    Copyright (C) 2018  Johan Dahlin < uni (at) johandahlin [dot] com >
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

"""Estimates a model using data from a OE model with Gaussian noise."""
import pickle
import stan
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import math
from scipy.io import loadmat, savemat

# OE model with Gaussian noise and horseshoe sparseness prior on the coefficients.

oe_code = """
data {
  int<lower=0> len_F;
  int<lower=0> len_B;
  int<lower=0> no_obs_est;
  real y_est[no_obs_est];
  real u_est[no_obs_est];
}
parameters {
    vector[len_B] b_coefs;
    vector[len_F] f_coefs;
    vector<lower=0>[len_B] b_coefs_hyperprior;
    vector<lower=0>[len_F] f_coefs_hyperprior;
    real<lower=0> sigmae;
    real<lower=0> shrinkage_param;
}
model {
    real w[no_obs_est];
    real err[no_obs_est];
    int max_order = max(len_F, len_B);
 
    for (i in 1:max_order) {
        w[i] = 0.0;
    }

    for (n in max_order:no_obs_est) {
        w[n]=0.0;

        for (i in 1:len_B) {
            w[n] += b_coefs[i] * u_est[n-i+1];
        }

        for (i in 1:len_F) {
            w[n] -= f_coefs[i] * w[n-i];
        }

       err[n] = y_est[n]-w[n];

       err[n] ~ normal(0,sigmae);
       
//      target+=0.5*(y_est[n]-w[n])^2/(sigmae^2) + log(sigmae^2);      
    
    }

    shrinkage_param ~ cauchy(0.0, 1.0);

    b_coefs_hyperprior ~ cauchy(0.0, 1.0);
    b_coefs ~ normal(0.0, b_coefs_hyperprior * shrinkage_param);

    f_coefs_hyperprior ~ cauchy(0.0, 1.0);
    f_coefs ~ normal(0.0, f_coefs_hyperprior * shrinkage_param);
    
    sigmae ~ cauchy(0.0, 0.1);

}
"""

np.random.seed(87655678)

data = loadmat("oe_ex_data1.mat")

y_est = data['y_estimation'].flatten()
u_est = data['u_estimation'].flatten()

F = data['A'].flatten()[1:]
B = data['B'].flatten()

vare = data['vare']

no_obs_est = len(y_est)

# Set initial values



def init_function():
    sigmae = math.sqrt(vare) * np.random.uniform(0.8, 1.2)
    b_coefs = B * np.random.uniform(0.8, 1.2, len(B))
    f_coefs = F * np.random.uniform(0.8, 1.2, len(F))
    output = {"b_coefs":b_coefs,"f_coefs":f_coefs,"sigmae": sigmae}
    return output

thisinit=[init_function(),init_function(),init_function(),init_function()]


stan_data = {
'len_B': len(B),
'len_F': len(F),
'no_obs_est': no_obs_est,
'y_est': y_est,
'u_est': u_est,
}

# Run Stan

posterior = stan.build(oe_code, data=stan_data, random_seed=87655678)
fit = posterior.sample(num_chains=4, num_warmup=4000, num_samples=2000, init=thisinit)

# Expore Results

df_stan_est_trace = fit.to_frame()
df_stan_est_trace.index.name=None

# Export output to feather files

df_stan_est_trace.to_feather('stan_oe_all.feather')

# Export results to Matlab

savemat('results.mat', {'results': df_stan_est_trace.to_dict('list')})

