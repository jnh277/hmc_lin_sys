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
// stan state space model
functions{
    real matrix_normal_lpdf(matrix y, matrix mu, matrix LSigma){
        int pdims[2] = dims(y);
        matrix[pdims[1],pdims[2]] error_sc = mdivide_left_tri_low(LSigma, y - mu);
        real p1 = -0.5*pdims[2]*(pdims[1]*log(2*pi()) + 2*sum(log(diagonal(LSigma))));
        real p2 = -0.5*sum(error_sc .* error_sc);
        return p1+p2;

    }
}

data {
    int<lower=0> no_obs_est;
    int<lower=0> no_states;
    row_vector[no_obs_est] y_est;
    row_vector[no_obs_est] u_est;
    real<lower=0> Ts;
}
transformed data{
    vector[no_states] Bs = rep_vector(0.0,no_states);
    Bs[no_states] = 1.0;
}
parameters {
    matrix[no_states,no_obs_est] h;         // hidden states
    row_vector[no_states] a_coefs;
    row_vector[no_states] C;
    real D;
    real<lower=0.0> r;                // measurement noise
    // components of process noise matrix
    // for now will just have diagonal components
    vector<lower=0,upper=pi()/2>[no_states] tauQ_unif;
    cholesky_factor_corr[no_states] LQcorr;

}
transformed parameters {
    matrix[no_states,no_states] A = rep_matrix(0.0,no_states,no_states);
    matrix[no_states,no_states] Ad;
    vector[no_states] Bd;
    vector<lower=0>[no_states] tauQ = 2.5 * tan(tauQ_unif);       // LQ diag scaling
    matrix[3,3] F;
    A[no_states,:] = a_coefs;
    for (i in 1:no_states-1) A[i,i+1] = 1.0;
    F = matrix_exp(append_row(append_col(A,Bs),[0,0,0]) * Ts);
    Ad = F[1:2,1:2];
    Bd = F[1:2,3];

}

model {
    LQcorr ~ lkj_corr_cholesky(2);
    r ~ normal(0.0, 1.0);
    h[:,1] ~ normal(0, 1.0);  // prior on initial state

    // parameter priors
    a_coefs ~ normal(0.0, 1.0);
    C ~ normal(0.0, 1.0);
    D ~ normal(0.0, 1.0);

    // state distributions
    target += matrix_normal_lpdf(h[:,2:no_obs_est] | Ad * h[:,1:no_obs_est-1] + Bd * u_est[1:no_obs_est-1], diag_pre_multiply(tauQ,LQcorr));

    // measurement distributions
    y_est ~ normal(C*h+D*u_est, r);
}
generated quantities {
    row_vector[no_obs_est] y_hat;
    cholesky_factor_cov[2] LQ;
    vector[no_states] B = Bs;
    y_hat = C*h+D*u_est;
    LQ = diag_pre_multiply(tauQ,LQcorr);


}


