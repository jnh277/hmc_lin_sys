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

// stan model for an RC circuit and its state

data {
    int<lower=0> no_obs_est;
    int<lower=0> no_obs_val;
    row_vector[no_obs_est] y_est;
    row_vector[no_obs_est] u_est;
    row_vector[no_obs_val] y_val;
    row_vector[no_obs_val] u_val;
//    real h0_val;
    real<lower=0> Ts;
}
parameters {
    real h0;                        // initial hidden state value
    real<lower=0.0> Cq;            // the capacitance
    real<lower=0.0> Rq;            // the resistance
    real<lower=0.0> q;                 // process noise
    real<lower=0.0> r;                // measurement noise
    real<lower=0.0> p0_sig;         // initial state sigma

}
transformed parameters {
    real A = -1/Rq/Cq;
    real Ad;
    real Bd;
    real C = 1/Cq;
    row_vector[no_obs_est] mu;
    row_vector[no_obs_est] P;
    matrix[2,2] F = matrix_exp([[A, 1.0],[0, 0]] * Ts);
    Ad = F[1,1];
    Bd = F[1,2];
    mu[1] = h0;
    P[1] = p0_sig * p0_sig;

    for (k in 2:no_obs_est){
        mu[k] = Ad*mu[k-1]+Bd*u_est[k-1];
        P[k] = Ad*P[k-1]*Ad + q*q;
    }


}
model {
    q ~ cauchy(0.0, 1.0);
    r ~ cauchy(0.0, 1.0);
    h0 ~ normal(0, p0_sig);          // prior on initial state
    p0_sig ~ cauchy(0, 1.0);

    // parameter priors
    Cq ~ inv_gamma(2.0, 1.0);
    Rq ~ inv_gamma(2.0, 1.0);

    // state distributions
    y_est ~ normal(C*mu, r+sqrt(P));
}
generated quantities {
    row_vector[no_obs_est] y_hat;
    y_hat = C*mu;

}

