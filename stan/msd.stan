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
// stan model for an RLC circuit and its states

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
    matrix[2,no_obs_est] h;         // hidden states
    real<lower=0.0> Mq;            // the capacitance
    real<lower=0.0> Dq;            // the resistance
    real<lower=0.0> Kq;             // inductance
    real<lower=0.0> q;                 // process noise
    real<lower=0.0> r;                // measurement noise

}
transformed parameters {
    matrix[2,2] Ad;
    vector[2] Bd;
    matrix[3,3] F = matrix_exp([[0, 1.0, 0.0],[-Kq/Mq, -Dq/Mq, 1.0/Mq],[0, 0, 0]] * Ts);
    Ad = F[1:2,1:2];
    Bd = F[1:2,3];
}

model {
    q ~ normal(0.0, 1.0);
    r ~ normal(0.0, 1.0);
    h[:,1] ~ normal(0, 1.0);  // prior on initial state

    // parameter priors
    Mq ~ inv_gamma(2.0, 1.0);
    Kq ~ inv_gamma(2.0, 1.0);
    Dq ~ inv_gamma(2.0, 1.0);

    // state distributions
    to_vector(h[:,2:no_obs_est]) ~ normal(to_vector(Ad * h[:,1:no_obs_est-1] + Bd * u_est[1:no_obs_est-1]),q);

    // measurement distributions
    y_est ~ normal(h[1,:], r);
}
generated quantities {
    row_vector[no_obs_est] y_hat;
    y_hat = h[1,:];

}


