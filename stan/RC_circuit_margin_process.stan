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
    vector[no_obs_est] y_est;
    vector[no_obs_est] u_est;
    real<lower=0> Ts;
}
parameters {
    vector[no_obs_est] h;
    real<lower=0.0> Cq;            // the capacitance
    real<lower=0.0> Rq;            // the resistance
    real<lower=0.0> q;                 // process noise
    real<lower=0.0> r;                // measurement noise

}
transformed parameters {
    matrix[no_obs_est,no_obs_est] Sk;       // upper sqrt of kernel
    row_vector[no_obs_est] mu;
    real A = -1/Rq/Cq;
    real Ad;
    real Bd;
    matrix[2,2] F = matrix_exp([[A, 1.0],[0, 0]] * Ts);
//    F = matrix_exp(append_row(append_col(to_vector(A),to_vector(1)),rep_matrix(0,1,2))*Ts);
    Ad = F[1,1];
    Bd = F[1,2];
    mu[1] = 0;

    for (i in 1:no_obs_est){
        Sk[1,i] = Ad^(i-1)*q / Cq;
        if (i > 1){
            mu[i] = (Ad*mu[i-1] + Bd*u_est[i-1])/Cq;
            for (j in i:no_obs_est){
                Sk[i,j] = Ad^(j-i)*q/Cq;
            }
        }
     }

}
model {
    q ~ cauchy(0.0, 1.0);
    r ~ cauchy(0.0, 1.0);
    h[1] ~ normal(0, 1.0);  // prior on initial state

    // parameter priors
    Cq ~ inv_gamma(2.0, 1.0);
    Rq ~ inv_gamma(2.0, 1.0);

    // measurement joint likelihood
    y_est ~ multi_normal_cholesky(mu/Cq, Sk');

}
//generated quantities {
//    vector[no_obs_est] y_hat;
//    y_hat = h/Cq;
//
//}

