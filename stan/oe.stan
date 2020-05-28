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
// stan Output error (i.e. discrete transfer function) model
// OE model with Gaussian noise and horseshoe sparseness prior on the coefficients.
// Using horsehoe prior, this estimates the error vector allowing one step ahead predictions
// at all points to be used

data {
  int<lower=0> output_order;
  int<lower=0> input_order;
  int<lower=0> no_obs_est;
  int<lower=0> no_obs_val;
  row_vector[no_obs_est] y_est;
  row_vector[no_obs_est] u_est;
  row_vector[no_obs_val] u_val;
  row_vector[no_obs_val] y_val;
}
transformed data {
    int<lower=0> max_order = max(output_order,input_order-1);
}
parameters {
    vector[input_order] b_coefs;
    vector[output_order] f_coefs;
    vector<lower=0>[input_order] b_coefs_hyperprior;
    vector<lower=0>[output_order] f_coefs_hyperprior;
    real<lower=0> shrinkage_param;
    real<lower=0> r;            // noise standard deviation
    real<lower=0> r2;
//    row_vector[max_order] e_init;
    row_vector[no_obs_est] ehat;
}
transformed parameters{
    row_vector[no_obs_est] yhat;
    yhat[1:max_order] = rep_row_vector(0.0,max_order);
    for (i in max_order+1:no_obs_est){
//        ehat[i] = y_est[i-output_order:i-1]*f_coefs + y_est[i] - u_est[i-input_order+1:i] * b_coefs
//              - ehat[i-output_order:i-1]*f_coefs;
        yhat[i] = u_est[i-input_order+1:i] * b_coefs - y_est[i-output_order:i-1]*f_coefs
                    + ehat[i-output_order:i-1]*f_coefs + ehat[i];
    }


}
model {
    // hyper priors
    shrinkage_param ~ cauchy(0.0, 1.0);
    b_coefs_hyperprior ~ cauchy(0.0, 1.0);
    f_coefs_hyperprior ~ cauchy(0.0, 1.0);

    // parameters
    b_coefs ~ normal(0.0, b_coefs_hyperprior * shrinkage_param);
    f_coefs ~ normal(0.0, f_coefs_hyperprior * shrinkage_param);
//    ehat ~ normal(0.0, r);

    // noise standard deviation
    r ~ cauchy(0.0, 1.0);
    r2 ~ cauchy(0.0, 1.0);

    // measurement likelihood
    ehat ~ normal(0.0, r);     // this includes the e_init prior
    y_est[max_order+1:no_obs_est] ~ normal(yhat[max_order+1:no_obs_est], r2);

}
generated quantities {
    row_vector[no_obs_val] y_hat_val = rep_row_vector(0.0,no_obs_val);
    row_vector[no_obs_val] y_hat_val2 = rep_row_vector(0.0,no_obs_val);
//    row_vector[no_obs_val] e_val = rep_row_vector(0.0,no_obs_val);
    for (i in max_order+1:no_obs_val){ // this isn't the best estimate of y_val as it doesnt have the error terms?
        y_hat_val[i] = u_val[i-input_order+1:i] * b_coefs
                - y_val[i-output_order:i-1] * f_coefs;// -e_val[i-output_order:i-1] * f_coefs;
//        e_val[i] = y_val[i] - y_hat_val[i];
    }
//    y_hat_val2[1:max_order] = y_val[1:max_order];
    for (n in (1+max_order):no_obs_val) {
            real foo = 0.0;
            foo -= y_hat_val2[n-output_order:n-1] * f_coefs;
            foo += u_val[n-input_order+1:n] * b_coefs;
            y_hat_val2[n] = foo;
    }
}

