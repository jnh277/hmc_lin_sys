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

// ARX model with Gaussian noise and horseshoe sparseness prior on the coefficients.

data {
    int<lower=0> output_order;
    int<lower=0> input_order;
    int<lower=0> no_obs_est;
    int<lower=0> no_obs_val;
    matrix[no_obs_est, output_order] est_obs_matrix;
    matrix[no_obs_est, input_order] est_input_matrix;
    matrix[no_obs_val, output_order] val_obs_matrix;
    matrix[no_obs_val, input_order] val_input_matrix;
    vector[no_obs_est] y_est;
}
parameters {
    vector[output_order] a_coefs;
    vector[input_order] b_coefs;
    vector<lower=0>[output_order] a_coefs_hyperprior;
    vector<lower=0>[input_order] b_coefs_hyperprior;
    real<lower=0> shrinkage_param;
    real<lower=0> sig_e;
}
model {
    a_coefs_hyperprior ~ cauchy(0.0, 1.0);
    b_coefs_hyperprior ~ cauchy(0.0, 1.0);
    shrinkage_param ~ cauchy(0.0, 1.0);
    a_coefs ~ normal(0.0, a_coefs_hyperprior * shrinkage_param);
    b_coefs ~ normal(0.0, b_coefs_hyperprior * shrinkage_param);

    sig_e ~ cauchy(0.0, 1.0);
    y_est ~ normal(-est_obs_matrix * a_coefs + est_input_matrix*b_coefs, sig_e);

}
generated quantities {
    vector[no_obs_val] y_hat;
    for (n in 1:no_obs_val) {
        y_hat[n] = val_input_matrix[n, :] * b_coefs-val_obs_matrix[n, :]*a_coefs;
    }

}

