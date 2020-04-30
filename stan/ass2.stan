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
// stan model for assignment 2 of mcha6100
functions{
    real matrix_normal_lpdf(matrix y, matrix mu, matrix LSigma){
        int pdims[2] = dims(y);
        matrix[pdims[1],pdims[2]] error_sc = mdivide_left_tri_low(LSigma, y - mu);
        real p1 = -0.5*pdims[2]*(pdims[1]*log(2*pi()) + 2*sum(log(diagonal(LSigma))));
        real p2 = -0.5*sum(error_sc .* error_sc);
        return p1+p2;

    }
    vector process_model(vector z, real u1, real u2, real m, real J, real l, real a, real r1, real r2){
        vector[5] dz;
        dz[1] = cos(z[3]) * z[4] / m;       // dx
        dz[2] = sin(z[3]) * z[4] / m;       // dy
        dz[3] = z[5]/(J+m*l^2);             // d\theta
        dz[4] = -r1*z[4]/m - m*l*z[5]^2/(J+m*l^2)^2 + u1 + u2;  // d p_1
        dz[5] = (l*z[4]-r2)*z[5]/(J+m*l^2) + u1*a - u2*a;
        return dz;
    }
    vector discrete_update(vector z, real u1, real u2, real m, real J, real l, real a, real r1, real r2, real Ts){
        vector[5] z_next;
        vector[5] dz;
        z_next = z;
        for (n in 1:20){
            dz = process_model(z, u1, u2, m, J, l, a, r1, r2);
            z_next = z_next + Ts/20 * dz;
        }
        return z_next;
    }
    matrix process_model_vec(matrix z, row_vector u1, row_vector u2, real m, real J, real l, real a, real r1, real r2){
        int pdims[2] = dims(z);
        matrix[pdims[1],pdims[2]] dz;
        dz[1,:] = cos(z[3,:]) .* z[4,:] / m;       // dx
        dz[2,:] = sin(z[3,:]) .* z[4,:] / m;       // dy
        dz[3,:] = z[5,:] / (J+m*l^2);             // d\theta
        dz[4,:] = -r1*z[4,:]/m - m*l^2*z[5,:] .* z[5,:] / (J+m*l^2)^2 + u1 + u2;  // d p_1
        dz[5,:] = (l*z[4,:]-r2) .* z[5,:] / (J+m*l^2) + u1*a - u2*a;
        return dz;
    }
    matrix discrete_update_vec(matrix z, row_vector u1, row_vector u2, real m, real J, real l, real a, real r1, real r2, real Ts){
        int pdims[2] = dims(z);
        matrix[pdims[1],pdims[2]] z_next;
        z_next = z;
        for (n in 1:10){
            z_next = z_next + Ts/10 * process_model_vec(z_next, u1, u2, m, J, l, a, r1, r2);
        }
        return z_next;
    }
    matrix rk4_update(matrix z, row_vector u1, row_vector u2, real m, real J, real l, real a, real r1, real r2, real Ts){
        int pdims[2] = dims(z);
//        matrix[pdims[1],pdims[2]] z_next;
        matrix[pdims[1],pdims[2]] k1;
        matrix[pdims[1],pdims[2]] k2;
        matrix[pdims[1],pdims[2]] k3;
        matrix[pdims[1],pdims[2]] k4;
        k1 = Ts * process_model_vec(z, u1, u2, m, J, l, a, r1, r2);
        k2 = Ts * process_model_vec(z+k1/2, u1, u2, m, J, l, a, r1, r2);
        k3 = Ts * process_model_vec(z+k2/2, u1, u2, m, J, l, a, r1, r2);
        k4 = Ts * process_model_vec(z+k3, u1, u2, m, J, l, a, r1, r2);

        return z + k1/6 + k2/3 + k3/3 + k4/6;
    }

}

data {
    int<lower=0> no_obs;
    matrix[3, no_obs] y;        // measurement [x, y, theta]
    row_vector[no_obs] u1;          // input 1
    row_vector[no_obs] u2;          // input 2
    real<lower=0> Ts;
    real a;                         // input lever
    real r1;                             // damping 1
    real r2;                             // damping 2
    vector[5] z0;                             // initial state guess
}
parameters {
    matrix[5,no_obs] h;         // hidden states
    real<lower=0.0> m;              // the mass
    real<lower=0.0> J;            // the inertia
    real<lower=0.0,upper=1.0> l;             // offset to center of gravity
//    real<lower=0.0> r;                // measurement noise
    // components of measurement noise
    vector<lower=0,upper=pi()/2>[3] tauR_unif;
    cholesky_factor_corr[3] LRcorr;
    // components of process noise matrix
    vector<lower=0,upper=pi()/2>[5] tauQ_unif;
    cholesky_factor_corr[5] LQcorr;

}
transformed parameters {
    matrix[5,no_obs] mu;
//    matrix[3,no_obs] yhat;
    vector<lower=0>[5] tauQ = 2.5 * tan(tauQ_unif);       // LQ diag scaling
    vector<lower=0>[3] tauR = 2.5 * tan(tauR_unif);       // LR diag scaling
    mu[:,1] = z0;
//    for (k in 1:no_obs-1) {
//        mu[:,k+1] = discrete_update(h[:,k], u1[k], u2[k], m, J, l, a, r1, r2, Ts);
//    }
    mu[:,2:no_obs] = discrete_update_vec(h[:,1:no_obs-1], u1[1:no_obs-1], u2[1:no_obs-1], m, J, l, a, r1, r2, Ts);
//    mu[:,2:no_obs] = rk4_update(h[:,1:no_obs-1], u1[1:no_obs-1], u2[1:no_obs-1], m, J, l, a, r1, r2, Ts);

}

model {
    LRcorr ~ lkj_corr_cholesky(2);
    LQcorr ~ lkj_corr_cholesky(2);
//    r ~ normal(0.0, 1.0);
    h[:,1] ~ normal(0, 1.0);  // prior on initial state

    // parameter priors
    m ~ cauchy(0,5);
    l ~ cauchy(0,1);
    J ~ cauchy(0,5);

    // state distributions
    target += matrix_normal_lpdf(h | mu, diag_pre_multiply(tauQ,LQcorr));

    // measurement distributions
//    y_est ~ normal(h[1,:], r);
    target += matrix_normal_lpdf( y| h[1:3,:], diag_pre_multiply(tauR,LRcorr));

}
generated quantities {
    cholesky_factor_cov[5] LQ;
    cholesky_factor_cov[3] LR;
    LQ = diag_pre_multiply(tauQ,LQcorr);
    LR = diag_pre_multiply(tauR,LRcorr);

}


