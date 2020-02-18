/*
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
*/
// stan Output error (i.e. discrete state space) model


// OE model with Gaussian noise and horseshoe sparseness prior on the coefficients.
// Using horsehoe prior

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
    row_vector[no_obs_est] yhat;
    vector[input_order] b_coefs;
    vector[output_order] f_coefs;
    vector<lower=0>[input_order] b_coefs_hyperprior;
    vector<lower=0>[output_order] f_coefs_hyperprior;
    real<lower=0> shrinkage_param;
    real<lower=0> r;            // noise standard deviation
    real<lower=0> q;
}
transformed parameters{
    vector[input_order] b_flip;
    vector[output_order] f_flip;
    row_vector[no_obs_est] mu = rep_row_vector(0.0,no_obs_est);
    row_vector[no_obs_est] w_est = rep_row_vector(0.0,no_obs_est);
    for (i in 1:output_order) f_flip[i] = f_coefs[output_order-1+1];
    for (i in 1:input_order) b_flip[i] = b_coefs[input_order-1+1];
    for (i in max_order+1:no_obs_est){
        mu[i] = u_est[i-input_order+1:i] * b_flip  -  yhat[i-output_order:i-1] * f_flip;
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
    q ~ cauchy(0.0, 1.0);

    // process model
    yhat[max_order+1:no_obs_est] ~ normal(yhat[max_order:no_obs_est-1],q);

    // measurement likelihood
    y_est[max_order+1:no_obs_est] ~ normal(yhat[max_order+1:no_obs_est], r);

}
generated quantities {
    row_vector[no_obs_val] y_hat_val = rep_row_vector(0.0,no_obs_val);

    for (i in max_order+1:no_obs_val){
        y_hat_val[i] = u_val[i-input_order+1:i] * b_coefs  - y_hat_val[i-output_order:i-1] * f_coefs;
    }
}

