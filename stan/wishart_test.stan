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
    int<lower=0> no_obs;
    int<lower=0> K;
//    vector[K] obs[no_obs];
    matrix[K,no_obs] obs;
//    vector[K] u[no_obs]; // input vectors
    matrix[K,no_obs] u;
}
//transformed data {
//}
parameters {
    cholesky_factor_corr[K] L_A;
    cholesky_factor_corr[K] L_Sigma;
    vector<lower=0,upper=pi()/2>[K] tau_unif_A;
    vector<lower=0,upper=pi()/2>[K] tau_unif_Sigma;
}
transformed parameters {
    vector<lower=0>[K] tau_A;           // prior scale
    vector<lower=0>[K] tau_Sigma;       // prior scale
    matrix[K,K] A;
    matrix[K,no_obs] mu;
//    vector[K] mu[no_obs];

    for (k in 1:K) tau_A[k] = 2.5 * tan(tau_unif_A[k]);
    for (k in 1:K) tau_Sigma[k] = 2.5 * tan(tau_unif_Sigma[k]);
    A = diag_pre_multiply(tau_A,L_A) * diag_pre_multiply(tau_A,L_A)';
//    for (i in 1:no_obs) mu[i] = A * u[i];
    mu = A*u;
}
model {

    L_A ~ lkj_corr_cholesky(2);

    L_Sigma ~ lkj_corr_cholesky(2);
//    obs ~ multi_normal_cholesky(mu, diag_pre_multiply(tau_Sigma,L_Sigma));
    target += matrix_normal_lpdf(obs | mu, diag_pre_multiply(tau_Sigma,L_Sigma));


}
generated quantities {
    cov_matrix[K] Sigma;
    Sigma = diag_pre_multiply(tau_Sigma,L_Sigma) * diag_pre_multiply(tau_Sigma,L_Sigma)';

}

