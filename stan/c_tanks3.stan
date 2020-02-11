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
    matrix[2,no_obs_est] y_est;
    row_vector[no_obs_est] u_est;
    matrix[2,no_obs_val] y_val;
    row_vector[no_obs_val] u_val;
    real<lower=0> Ts;
}
parameters {
    matrix[3,no_obs_est] h;         // hidden states
    vector<lower=0.0>[3] Cq;            // the capacitances
    vector<lower=0.0>[4] Rq;            // the resistances
    real<lower=0.0> q;                 // process noise
    real<lower=0.0> r;                // measurement noise

}
transformed parameters {
    matrix[3,3] Ad;
    vector[3] Bd;
    matrix[2,3] C = [[1/Cq[1],0,0],[0,0,1/Cq[3]]];
    matrix[4,4] Fp =[[-(1/Rq[1]+1/Rq[2])/Cq[1], 1/Rq[1]/Cq[2], 1/Rq[2]/Cq[2], 1],[1/Rq[1]/Cq[1], -(1/Rq[1]+1/Rq[3])/Cq[2], 1/Rq[3]/Cq[3], 0],[1/Rq[2]/Cq[1], 1/Rq[3]/Cq[2], -(1/Rq[2]+1/Rq[3]+1/Rq[4])/Cq[3], 0],[0, 0, 0, 0]] * Ts;
    matrix[4,4] F = matrix_exp(Fp);
    Ad = F[1:3,1:3];
    Bd = F[1:3,4];
}

model {
    q ~ cauchy(0.0, 1.0);
    r ~ cauchy(0.0, 1.0);
    h[:,1] ~ normal(0, 1.0);  // prior on initial state

    // parameter priors
    Cq ~ inv_gamma(2.0, 1.0);
    Rq ~ inv_gamma(2.0, 1.0);

    // state distributions
    to_vector(h[:,2:no_obs_est]) ~ normal(to_vector(Ad * h[:,1:no_obs_est-1] + Bd * u_est[1:no_obs_est-1]),q);

    // measurement distributions
    to_vector(y_est) ~ normal(to_vector(C*h), r);
}
generated quantities {
    matrix[2,no_obs_est] y_hat;
    y_hat = C*h;

}


