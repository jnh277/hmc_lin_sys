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
// stan model for estimating msd by marginalising over states

data {
    int<lower=0> no_obs_est;
    int<lower=0> no_obs_val;
    row_vector[no_obs_est] y_est;
    row_vector[no_obs_est] u_est;
    row_vector[no_obs_val] y_val;
    row_vector[no_obs_val] u_val;
    real<lower=0> Ts;
}
parameters {
    vector[2] h0;         // unknown initial states
    real<lower=0.0> h0_sig;     // unknown initial state variance
    real<lower=0.0> Mq;            // the capacitance
    real<lower=0.0> Kq;            // the resistance
    real<lower=0.0> Dq;             // inductance
    real<lower=0.0> q;                 // process noise
    real<lower=0.0> r;                // measurement noise

}
transformed parameters {
    matrix[2, no_obs_est] mu_pred;
    matrix[2,2] P_pred[no_obs_est];
    matrix[2, no_obs_est] mu_corr;
    matrix[2,2] P_corr[no_obs_est];
    vector[2] K;
    row_vector[2] C = [1, 0];
    row_vector[no_obs_est] Rk;
    matrix[2,2] Ad;
    vector[2] Bd;
    matrix[3,3] F = matrix_exp([[0, 1.0, 0.0],[-Kq/Mq, -Dq/Mq, 1.0/Mq],[0, 0, 0]] * Ts);
    Ad = F[1:2,1:2];
    Bd = F[1:2,3];
    mu_pred[:,1] = h0;
    P_pred[1] = [[h0_sig*h0_sig, 0],[0, h0_sig*h0_sig]];
    Rk[1] = r*r;
    K = P_pred[1]*C'/(C*P_pred[1]*C'+r*r);
    mu_corr[:,1]  =mu_pred[:,1] + K*(y_est[1] - C*mu_pred[:,1]);
    P_corr[1] = (diag_matrix(rep_vector(1.0,2)) - K*C)*P_pred[1]*(diag_matrix(rep_vector(1.0,2)) - K*C)'+K*r*r*K';

    for (k in 2:no_obs_est){
        mu_pred[:,k] = Ad*mu_corr[:,k-1] + Bd*u_est[k-1];
        P_pred[k] = Ad*P_corr[k-1]*Ad' + diag_matrix(rep_vector(q,2));
        Rk[k] = C*P_pred[k]*C';
        K = P_pred[k-1]*C'/(C*P_pred[k-1]*C'+r*r);
        mu_corr[:,k] = mu_pred[:,k] + K*(y_est[k] - C*mu_pred[:,k]);
        P_corr[k] = (diag_matrix(rep_vector(1.0,2)) - K*C)*P_pred[k]*(diag_matrix(rep_vector(1.0,2)) - K*C)'+K*r*r*K';
    }
}

model {
    q ~ normal(0.0, 1.0);
    r ~ normal(0.0, 1.0);
    h0 ~ normal(0, h0_sig);  // prior on initial state
    h0_sig ~ cauchy(0, 1.0);

    // parameter priors
    Mq ~ inv_gamma(2.0, 1.0);
    Kq ~ inv_gamma(2.0, 1.0);
    Dq ~ inv_gamma(2.0, 1.0);

    // measurement distributions
    y_est ~ normal(C*mu_pred, sqrt(Rk)+r);
}
generated quantities {
    row_vector[no_obs_est] y_hat;
    y_hat = C*mu_pred;

}


