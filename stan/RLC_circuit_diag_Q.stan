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
// stan model for an RLC circuit and its states with a full covariance matrix

//functions{
////    real vector_normal_lpdf(matrix y, matrix mu, vector sigma_q){
////        int pdims[2] = dims(y);
////        vector[pdims[1]] sc_e = diagonal((y - mu) * (y-mu)')./ (sigma_q .* sigma_q);
//////        real p1;
//////        p1 = -pdims[1]*0.5*log(2*pi()) - 0.5 * sum(log(sigma_q .* sigma_q));
////        return -pdims[1]*0.5*log(2*pi()) - 0.5 * sum(log(sigma_q .* sigma_q)) - 0.5 * sum(sc_e);
////    }
//    real vector_normal_lpdf(matrix y, matrix mu, vector sigma_q){
//        int pdims[2] = dims(y);
//        vector[pdims[1]] sc_e = diagonal((y - mu) * (y-mu)')./ (sigma_q .* sigma_q);
////        real p1;
////        p1 = -pdims[1]*0.5*log(2*pi()) - 0.5 * sum(log(sigma_q .* sigma_q));
//        return -pdims[1]*0.5*log(2*pi()) - 0.5 * sum(log(sigma_q .* sigma_q)) - 0.5 * sum(sc_e);
//    }
//}

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
    real<lower=0.0> Cq;            // the capacitance
    real<lower=0.0> Rq;            // the resistance
    real<lower=0.0> Lq;             // inductance
    real<lower=0.0> q;                 // process noise
    real<lower=0.0> r;                // measurement noise
    vector<lower=0.0>[2] diag_q;             // diagonal process noise

}
transformed parameters {
    matrix[2,no_obs_est] mu;
    matrix[2,2] Ad;
    vector[2] Bd;
    matrix[3,3] F = matrix_exp([[-Rq/Lq, -1/Cq, 1.0],[1/Lq, 0, 0.0],[0, 0, 0]] * Ts);
    Ad = F[1:2,1:2];
    Bd = F[1:2,3];
    mu = Ad*h+Bd*u_est;
}

model {

    diag_q ~ cauchy(0.0, 1.0); // diagonal process noise
    r ~ cauchy(0.0, 1.0);    // measurement noise
    h[:,1] ~ normal(0, 1.0);  // prior on initial state

    // parameter priors
    Cq ~ inv_gamma(2.0, 1.0);
    Rq ~ inv_gamma(2.0, 1.0);
    Lq ~ inv_gamma(2.0, 1.0);

    // state distributions


//    target += vector_normal_lpdf(h[:,2:no_obs_est] | h[:,1:no_obs_est-1]+Bd*u_est[1:no_obs_est-1], diag_q);

    to_vector(h[1,2:no_obs_est]) ~ normal(to_vector(mu[1,1:no_obs_est-1]), diag_q[1]);
    to_vector(h[2,2:no_obs_est]) ~ normal(to_vector(mu[2,1:no_obs_est-1]), diag_q[2]);
    y_est ~ normal(h[2,:]/Cq, r);
}
generated quantities {
    row_vector[no_obs_est] y_hat;
    y_hat = h[2,:]/Cq;

}


