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
//transformed data {
//    vector[no_states] vs;
//    vector[no_outputs] vy;
//    cov_matrix[no_states] Sig_s;
//    cov_matrix[no_outputs] Sig_y;
//    Sig_s = diag_matrix(rep_vector(1.0/no_states,no_states));
//    Sig_y = diag_matrix(rep_vector(1.0/no_outputs,no_outputs));
//
//
//}
parameters {
    matrix[no_outputs,no_states] C;
    matrix[no_states,no_inputs] B;
    matrix[no_outputs,no_states] D;
    cholesky_factor_corr[no_states] L_negA;
    cholesky_factor_corr[no_states] L_Q;
    cholesky_factor_corr[no_outputs] L_R;
//    vector<lower=0.000001,upper=pi()/2>[no_states] tau_unif_A;
//    vector<lower=0.000001,upper=pi()/2>[no_states] tau_unif_Q;
//    vector<lower=0.000001,upper=pi()/2>[no_outputs] tau_unif_R;
    vector<lower=0.000001>[no_states] tau_A;
    vector<lower=0.000001>[no_states] tau_Q;
    vector<lower=0.000001>[no_outputs] tau_R;

    vector[no_states] h[no_obs_est];
//    real<lower=0> b_hypers[no_states*no_inputs];
//    real<lower=0> c_hypers[no_outputs*no_states];
//    real<lower=0> d_hypers[no_outputs*no_inputs];
//    real<lower=0> shrinkage_param;
}
transformed parameters {
//    matrix[no_states+no_inputs,no_states+no_inputs] F;
    matrix[no_states,no_states] negA;
//    matrix[no_states,no_states] Ad;
//    matrix[no_states,no_inputs] Bd;
//    vector[no_states] tau_A;
//    vector[no_states] tau_Q;
//    vector[no_outputs] tau_R;


//    for (k in 1:no_states) tau_A[k] = 2.5 * tan(tau_unif_A[k]);
//    for (k in 1:no_states) tau_Q[k] = 2.5 * tan(tau_unif_Q[k]);
//    for (k in 1:no_outputs) tau_R[k] = 2.5 * tan(tau_unif_R[k]);

    negA = diag_pre_multiply(tau_A,L_negA) * diag_pre_multiply(tau_A,L_negA)';

//    F = matrix_exp(append_row(append_col(-negA,B),rep_matrix(0,no_inputs,no_states+no_inputs))*Ts);
//    Ad = F[1:no_states,1:no_states];
//    Bd = F[1:no_states,no_states+1:no_states+no_inputs];
    // struggling with matrix exponential (maybe)
}
model {
    vector[no_states] mu1[no_obs_est];
    vector[no_outputs] mu2[no_obs_est];
    // prior on scales
    tau_A ~ normal(0.0, 1.0);
    tau_Q ~ normal(0.0, 1.0);
    tau_R ~ normal(0.0, 1.0);

    // prior on noise terms
    L_Q ~ lkj_corr_cholesky(2);
    L_R ~ lkj_corr_cholesky(2);

    // prior on initial state
    h[1] ~ normal(0.0, 1.0);

    // parameter priors
    L_negA ~ lkj_corr_cholesky(2);
    to_vector(B) ~ cauchy(0.0, 1.0);
    to_vector(C) ~ cauchy(0.0, 1.0);
    to_vector(D) ~ cauchy(0.0, 1.0);

    // measurement model
    for (t in 2:no_obs_est)
        mu1[t] =  h[t] + Ts*(-negA*h[t]+B*u_est[t]);

    for (t in 1:no_obs_est)
        mu2[t] =  C*h[t] + D*u_est[t];

    h[2:no_obs_est] ~ multi_normal_cholesky(mu1[2:no_obs_est], diag_pre_multiply(tau_Q,L_Q));
    y_est ~ multi_normal_cholesky(mu2, diag_pre_multiply(tau_R,L_R));

}
generated quantities {
    vector[no_outputs] y_hat[no_obs_est];
    matrix[no_states,no_states] A = -negA;
    matrix[no_outputs,no_outputs] R = diag_pre_multiply(tau_R,L_R) * diag_pre_multiply(tau_R,L_R)';
    matrix[no_states,no_states] Q = diag_pre_multiply(tau_Q,L_Q) * diag_pre_multiply(tau_Q,L_Q)';
    for (t in 1:no_obs_est)
        y_hat[t] =  C*h[t] + D*u_est[t];

}

