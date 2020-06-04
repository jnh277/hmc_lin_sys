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
// stan Output error (i.e. discrete transfer function) model


// OE model with Gaussian noise and horseshoe sparseness prior on the coefficients.
// Using horsehoe prior, using open loop propagation (which doesnt work)

data {
  int<lower=0> no_obs_est;
  int<lower=0> no_obs_val;
  row_vector[no_obs_est] y_est;
  row_vector[no_obs_est] u_est;
  row_vector[no_obs_val] u_val;
}
transformed data {
    int<lower=0> input_order = 3;
    int<lower=0> output_order = 2;
    int<lower=0> max_order = max(output_order,input_order-1);
}
parameters {
    vector[input_order] b_coefs;
    real mags_a;
    real phase_a;
    vector<lower=0>[input_order] b_coefs_hyperprior;
    vector<lower=0>[output_order] fb_coefs_hyperprior;
    real<lower=0> shrinkage_param;
    real<lower=0> r;            // noise standard deviation
}
transformed parameters{
    vector[input_order] b_coefs;
    vector[output_order] f_coefs;
    row_vector[no_obs_est] mu = rep_row_vector(0.0,no_obs_est);
    f_coefs[1] = fb_coefs[1]*fb_coefs[2];
    f_coefs[2] = fb_coefs[1] + fb_coefs[2];
    b_coefs[1] = bb_coefs[2]*fb_coefs[2]+ bb_coefs[3]*fb_coefs[1] +bb_coefs[1]*fb_coefs[1]*fb_coefs[2];
    b_coefs[2] = bb_coefs[2]+bb_coefs[3] + bb_coefs[1]*(fb_coefs[1]+fb_coefs[2]);
    b_coefs[3] = bb_coefs[1];

    for (i in max_order+1:no_obs_est){
        mu[i] = u_est[i-input_order+1:i] * b_coefs  -  mu[i-output_order:i-1] * f_coefs;
    }


}
model {
    // hyper priors
    shrinkage_param ~ cauchy(0.0, 1.0);
    bb_coefs_hyperprior ~ cauchy(0.0, 1.0);
    fb_coefs_hyperprior ~ cauchy(0.0, 1.0);

    // parameters
    bb_coefs ~ normal(0.0, bb_coefs_hyperprior * shrinkage_param);
    fb_coefs ~ normal(0.0, fb_coefs_hyperprior * shrinkage_param);

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

