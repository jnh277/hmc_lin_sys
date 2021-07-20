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

"""Estimates an OE model using data with Gaussian noise and horeshoe priors."""

import stan
import numpy as np
from scipy.io import loadmat
import pickle
from pathlib import Path


# specific data path
data_path = 'data/example3_oe.mat'
input_order = 4
output_order = 3

def run_oe_hmc(data_path, input_order, output_order, hot_start=False, iter=6000, OL=False):
    """ Input order gives the terms b_0 * u_k + b_1 * u_{k-1} + .. + b_{input_order-1} * u_{k-input_order+1}"""
    """ Output order gives the terms a_0 * y_{k-1} + ... + a_{output_order-1}*y_{k-output_order} """
    """ hot_start=True will initialise at maximum likelihood results"""
    data = loadmat(data_path)

    y_est = data['y_estimation'].flatten()
    u_est = data['u_estimation'].flatten()
    y_val = data['y_validation'].flatten()
    u_val = data['u_validation'].flatten()

    stan_data = {'input_order': int(input_order),
                 'output_order': int(output_order),
                 'no_obs_est': len(y_est),
                 'y_est': y_est,
                 'y_val': y_val,
                 'u_est':u_est,
                 'u_val':u_val,
                 'no_obs_val': len(y_val),
                 }

    # Run Stan
    if hot_start:
        def init_function():
            f_true = data['f_ml'].flatten()[1:output_order+1]
            b_true = data['b_ml'].flatten()
            sig_e = data['sig_e'].flatten()
            output = dict(f_coefs=np.flip(f_true) * np.random.uniform(0.8, 1.2, len(f_true)),
                          b_coefs=np.flip(b_true) * np.random.uniform(0.8, 1.2, len(b_true)),
                          r=(sig_e* np.random.uniform(0.8, 1.2))[0],
                          )
            return output
    else:
        def init_function():
            sig_e = data['sig_e'].flatten()
            output = dict(r=(sig_e * np.random.uniform(0.8, 1.2))[0],
                          )
            return output

    if OL:
        model_code = """
        data {
              int<lower=0> output_order;
              int<lower=0> input_order;
              int<lower=0> no_obs_est;
              int<lower=0> no_obs_val;
              row_vector[no_obs_est] y_est;
              row_vector[no_obs_est] u_est;
              row_vector[no_obs_val] u_val;
            }
            transformed data {
                int<lower=0> max_order = max(output_order,input_order-1);
            }
            parameters {
                vector[input_order] b_coefs;
                vector[output_order] f_coefs;
                vector<lower=0>[input_order] b_coefs_hyperprior;
                vector<lower=0>[output_order] f_coefs_hyperprior;
                real<lower=0> shrinkage_param;
                real<lower=0> r;            // noise standard deviation
            }
            transformed parameters{
                row_vector[no_obs_est] mu = rep_row_vector(0.0,no_obs_est);
                for (i in max_order+1:no_obs_est){
                    mu[i] = u_est[i-input_order+1:i] * b_coefs  -  mu[i-output_order:i-1] * f_coefs;
                }
            
            
            }
            model {
                // hyper priors
                shrinkage_param ~ cauchy(0.0, 1.0);
                b_coefs_hyperprior ~ cauchy(0.0, 1.0);
                f_coefs_hyperprior ~ cauchy(0.0, 1.0);
            
                // parameters
                b_coefs ~ normal(0.0, b_coefs_hyperprior * shrinkage_param);
                f_coefs ~ normal(0.0, f_coefs_hyperprior * shrinkage_param);
            
                // noise standard deviation
                r ~ cauchy(0.0, 1.0);
            
                // measurement likelihood
                y_est[max_order+1:no_obs_est] ~ normal(mu[max_order+1:no_obs_est], r);
            
            }
            generated quantities {
                 row_vector[no_obs_val] y_hat_val = rep_row_vector(0.0,no_obs_val);
             
                 for (i in max_order+1:no_obs_val){
                     y_hat_val[i] = u_val[i-input_order+1:i] * b_coefs  - y_hat_val[i-output_order:i-1] * f_coefs;
                 }
             }
        """

    else:
        model_code = """
                    data {
                          int<lower=0> output_order;
                          int<lower=0> input_order;
                          int<lower=0> no_obs_est;
                          int<lower=0> no_obs_val;
                          row_vector[no_obs_est] y_est;
                          row_vector[no_obs_est] u_est;
                          row_vector[no_obs_val] u_val;
                          row_vector[no_obs_val] y_val;
                        }
                        transformed data {
                            int<lower=0> max_order = max(output_order,input_order-1);
                        }
                        parameters {
                            vector[input_order] b_coefs;
                            vector[output_order] f_coefs;
                            vector<lower=0>[input_order] b_coefs_hyperprior;
                            vector<lower=0>[output_order] f_coefs_hyperprior;
                            real<lower=0> shrinkage_param;
                            real<lower=0> r;            // noise standard deviation
                            real<lower=0> r2;
                        //    row_vector[max_order] e_init;
                            row_vector[no_obs_est] ehat;
                        }
                        transformed parameters{
                            row_vector[no_obs_est] yhat;
                            yhat[1:max_order] = rep_row_vector(0.0,max_order);
                            for (i in max_order+1:no_obs_est){
                        //        ehat[i] = y_est[i-output_order:i-1]*f_coefs + y_est[i] - u_est[i-input_order+1:i] * b_coefs
                        //              - ehat[i-output_order:i-1]*f_coefs;
                                yhat[i] = u_est[i-input_order+1:i] * b_coefs - y_est[i-output_order:i-1]*f_coefs
                                            + ehat[i-output_order:i-1]*f_coefs + ehat[i];
                            }
                        
                        
                        }
                        model {
                            // hyper priors
                            shrinkage_param ~ cauchy(0.0, 1.0);
                            b_coefs_hyperprior ~ cauchy(0.0, 1.0);
                            f_coefs_hyperprior ~ cauchy(0.0, 1.0);
                        
                            // parameters
                            b_coefs ~ normal(0.0, b_coefs_hyperprior * shrinkage_param);
                            f_coefs ~ normal(0.0, f_coefs_hyperprior * shrinkage_param);
                        //    ehat ~ normal(0.0, r);
                        
                            // noise standard deviation
                            r ~ cauchy(0.0, 1.0);
                            r2 ~ cauchy(0.0, 1.0);
                        
                            // measurement likelihood
                            ehat ~ normal(0.0, r);     // this includes the e_init prior
                            y_est[max_order+1:no_obs_est] ~ normal(yhat[max_order+1:no_obs_est], r2);
                        
                        }
        """

    posterior = stan.build(model_code, data=stan_data)
    init = [init_function(),init_function(),init_function(),init_function()]
    traces = posterior.sample(num_chains=4, num_warmup=4000, num_samples=2000, init=init)

    return traces
