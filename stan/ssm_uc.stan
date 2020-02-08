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


data {
    int<lower=0> no_obs_est;
    int<lower=0> no_obs_val;
    int<lower=0> no_states;
    int<lower=0> no_outputs;
    int<lower=0> no_inputs;
    real<lower=0> Ts;
    vector[no_outputs] y_est[no_obs_est];
    vector[no_inputs] u_est[no_obs_est];
    vector[no_outputs] y_val[no_obs_val];
    vector[no_inputs] u_val[no_obs_val];
}
parameters {
    matrix[no_states,no_states] Ad;
    matrix[no_outputs,no_states] C;
    matrix[no_states,no_inputs] Bd;
    matrix[no_outputs,no_states] D;
    vector<lower=0>[no_states*no_states] Ad_hyper;
    vector<lower=0>[no_outputs*no_states] C_hyper;
    vector<lower=0>[no_states*no_inputs] Bd_hyper;
    vector<lower=0>[no_outputs*no_states] D_hyper;
    real<lower=0> shrinkage_param;
    cholesky_factor_corr[no_states] L_Q;
    cholesky_factor_corr[no_outputs] L_R;
    vector<lower=0.000001,upper=pi()/2>[no_states] tau_unif_Q;
    vector<lower=0.000001,upper=pi()/2>[no_outputs] tau_unif_R;
    vector[no_states] h[no_obs_est];

}
transformed parameters {
    vector[no_states] tau_Q;
    vector[no_outputs] tau_R;
    for (k in 1:no_states) tau_Q[k] = 2.5 * tan(tau_unif_Q[k]);
    for (k in 1:no_outputs) tau_R[k] = 2.5 * tan(tau_unif_R[k]);
}
model {
    vector[no_states] mu1[no_obs_est];
    vector[no_outputs] mu2[no_obs_est];

    // prior on noise terms
    L_Q ~ lkj_corr_cholesky(2);
    L_R ~ lkj_corr_cholesky(2);

    // prior on initial state
    h[1] ~ normal(0.0, 1.0);

    // parameter priors
    Ad_hyper ~ cauchy(0.0, 1);
    Bd_hyper ~ cauchy(0.0, 1);
    C_hyper ~ cauchy(0.0, 1);
    D_hyper ~ cauchy(0.0, 1);
    shrinkage_param ~ cauchy(0.0, 1);

    to_vector(Ad) ~ normal(0.0, Ad_hyper .* Ad_hyper * shrinkage_param^2);
    to_vector(Bd) ~ normal(0.0, Bd_hyper .* Bd_hyper * shrinkage_param^2);
    to_vector(C) ~ normal(0.0, Bd_hyper .* Bd_hyper * shrinkage_param^2);
    to_vector(D) ~ normal(0.0, Bd_hyper .* Bd_hyper * shrinkage_param^2);

    // measurement model
    for (t in 2:no_obs_est)
        mu1[t] =  Ad*h[t]+Bd*u_est[t];

    for (t in 1:no_obs_est)
        mu2[t] =  C*h[t] + D*u_est[t];

    h[2:no_obs_est] ~ multi_normal_cholesky(mu1[2:no_obs_est], diag_pre_multiply(tau_Q,L_Q));
    y_est ~ multi_normal_cholesky(mu2, diag_pre_multiply(tau_R,L_R));

}
generated quantities {
    vector[no_outputs] y_hat[no_obs_est];
    for (t in 1:no_obs_est)
        y_hat[t] =  C*h[t] + D*u_est[t];

}

