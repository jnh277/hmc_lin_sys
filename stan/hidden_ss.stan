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
}
parameters {
    vector[no_obs_est] h;
    real<lower=0,upper=1> a;    // in actual fact this could go to negative one
    real<lower=0> b;            // b and c have an unobservable symmetry
    real<lower=0> c;
    real d;
    real<lower=0> d_hyper;
    real<lower=0> r;
    real<lower=0> q;


//    vector<lower=0>[input_order] b_coefs_hyperprior;
//    real<lower=0> shrinkage_param;
//    real<lower=0> sig_e;
}
model {
    q ~ cauchy(0.0, 1.0);
    r ~ cauchy(0.0, 1.0);
    h[1] ~ cauchy(0, 1.0);  // prior on initial state

    // parameter priors
    a ~ beta(2,2);
//    a ~ cauchy(0, 1.0);
    b ~ cauchy(0, 1.0); // would a horseshoe prior on these be better
    c ~ cauchy(0, 1.0);
    d_hyper ~ cauchy(0,1.0);
    d ~ normal(0, d_hyper^2); // it probably is for d which can often be zero

    h[2:no_obs_est] ~ normal(a*h[1:no_obs_est-1]+b*u_est[1:no_obs_est-1], r);
    y_est ~ normal(c*h + d*u_est, r);
}
generated quantities {
    vector[no_obs_est] y_hat;
    y_hat = c*h + d*u_est;

}

