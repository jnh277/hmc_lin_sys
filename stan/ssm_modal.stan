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
// stan state space model in modal form
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
    int<lower=0> no_obs_val;
    int<lower=0> no_states;
    row_vector[no_obs_est] y_est;
    row_vector[no_obs_est] u_est;
    row_vector[no_obs_val] y_val;
    row_vector[no_obs_val] u_val;
    real<lower=0> Ts;
}
parameters {
    matrix[2,no_obs_est] h;         // hidden states
    vector<lower=0.0>[2] p_real;
    vector[1] p_complex;
    vector[2] B;
    row_vector[2] C;
    real D;
    real<lower=0.0> r;                // measurement noise
    // components of process noise matrix
    // for now will just have diagonal components
    vector<lower=0,upper=pi()/2>[2] tauQ_unif;
    cholesky_factor_corr[2] LQcorr;

}
transformed parameters {
    matrix[2,2] Ad;
    vector[2] Bd;
    vector<lower=0>[2] tauQ = 2.5 * tan(tauQ_unif);       // LQ diag scaling
    matrix[3,3] F;
    matrix[2,2] A = -diag_matrix(p_real);
    A[1,2] = p_complex[1];
    A[2,1] = -p_complex[1];
    F = matrix_exp(append_row(append_col(A,B),[0,0,0]) * Ts);
    Ad = F[1:2,1:2];
    Bd = F[1:2,3];

}

model {
    LQcorr ~ lkj_corr_cholesky(2);
    r ~ normal(0.0, 1.0);
    h[:,1] ~ normal(0, 1.0);  // prior on initial state

    // parameter priors
    p_real ~ normal(0.0, 1.0);
    p_complex ~ normal(0.0, 1.0);
    C ~ normal(0.0, 1.0);
    D ~ normal(0.0, 1.0);

    // state distributions
    target += matrix_normal_lpdf(h[:,2:no_obs_est] | Ad * h[:,1:no_obs_est-1] + Bd * u_est[1:no_obs_est-1], diag_pre_multiply(tauQ,LQcorr));

    // measurement distributions
    y_est ~ normal(h[1,:], r);
}
generated quantities {
    row_vector[no_obs_est] y_hat;
    cholesky_factor_cov[2] LQ;
    y_hat = h[1,:];
    LQ = diag_pre_multiply(tauQ,LQcorr);

}


