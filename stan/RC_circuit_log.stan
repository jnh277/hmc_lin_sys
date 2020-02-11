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
    vector[no_obs_est] y_est;
    vector[no_obs_est] u_est;
    vector[no_obs_val] y_val;
    vector[no_obs_val] u_val;
    real<lower=0> Ts;
}
parameters {
    vector[no_obs_est] h;
    real log_Cq;            // log of the capacitance
    real log_Rq;            // log of the resistance
    real<lower=0> q;                 // process noise
    real<lower=0> r;                // measurement noise

}
transformed parameters {
    real A = -1/exp(log_Cq)/exp(log_Rq);
    real Ad;
    real Bd;
    matrix[2,2] F = matrix_exp([[A, 1.0],[0, 0]] * Ts);
//    F = matrix_exp(append_row(append_col(to_vector(A),to_vector(1)),rep_matrix(0,1,2))*Ts);
    Ad = F[1,1];
    Bd = F[1,2];
}
model {
    q ~ cauchy(0.0, 1.0);
    r ~ cauchy(0.0, 1.0);
    h[1] ~ normal(0, 1.0);  // prior on initial state

    // parameter priors
    log_Cq ~ normal(0, 1.0);
    log_Rq ~ normal(0, 1.0);

    // state distributions
    h[2:no_obs_est] ~ normal(Ad*h[1:no_obs_est-1]+Bd*u_est[1:no_obs_est-1], q);
    y_est ~ normal(h/exp(log_Cq), r);
}
generated quantities {
    vector[no_obs_est] y_hat;
    y_hat = h/exp(log_Cq);

}

