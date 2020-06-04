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

// FIR model with Gaussian noise and a tuned-correlated (TC) kernel prior.

data {
    int<lower=0> input_order;
    int<lower=0> no_obs_est;
    int<lower=0> no_obs_val;
    vector[no_obs_est] y_est;
    matrix[no_obs_est, input_order] est_input_matrix;
    matrix[no_obs_val, input_order] val_input_matrix;
}
transformed data {
  vector[input_order] mu = rep_vector(0, input_order);
}
parameters {
    vector[input_order] b_coefs;
    real<lower=0> sig_e;
    real<lower=0> c;
    real<lower=0.01,upper=1> lam;
}
transformed parameters{

}
model {
    matrix [input_order, input_order] L_K;
    vector[2] tcpart;
    matrix [input_order, input_order] K = rep_matrix(0.0, input_order, input_order);
    c ~ gamma(0.01, 0.01);
    lam ~ gamma(0.01, 0.01);

    for (i in 1:input_order){
        for (j in 1:i) {
            tcpart[1] = lam^i;
            tcpart[2] = lam^j;
            K[i, j] = c * min(tcpart);
            K[j, i] = K[i, j];
        }
    }

    K = 0.5 * (K + K');
    L_K = cholesky_decompose(K);
    b_coefs ~ multi_normal_cholesky(mu, L_K);

    sig_e ~ cauchy(0.0, 1.0);
    y_est ~ normal(est_input_matrix*b_coefs, sig_e);
}
generated quantities {
    vector[no_obs_val] y_hat;
    for (n in 1:no_obs_val) {
        y_hat[n] = val_input_matrix[n, :] * b_coefs;
    }

}

