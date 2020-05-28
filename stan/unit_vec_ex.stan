data {
    int<lower=0> N;
    vector[N] y;
}
transformed data {
    unit_vector[2] y_xy[N];
    for (i in 1:N){
        y_xy[i,1] = cos(y[i]);
        y_xy[i,2] = sin(y[i]);
    }
}

parameters {
    real<lower=-1.0,upper=1.0> a;
    real<lower=0.0> r;
    real<lower=0.0> q1;
    real<lower=0.0> q2;
    unit_vector[2] xy[N];
    vector[2] w_xy[N];
//    vector[N] theta;
//    vector[N] omega;
}
//transformed parameters{
////    unit_vector[2] w_xy[N];
////    for (i in 1:N){
////        w_xy[i,1] = cos(omega[i]);
////        w_xy[i,2] = sin(omega[i]);
////    }
//
//}
model {
    // noise stds priors
    r ~ cauchy(0,1.0);
    q1 ~ cauchy(0, 0.1);
    q2 ~ cauchy(0, 1);

    // prior on parameter
    a ~ cauchy(0, 1.0);

    // initial state prior
//    omega[1] ~ cauchy(0,5);
//    theta[1] ~ cauchy(0,5);

    //
//    omega[2:N] ~ normal(omega[1:N-1], q);
//    theta[2:N] ~ normal(theta[1:N-1] + omega[1:N-1], q);

    for (i in 2:N){
//         state likelihood
        xy[i,:] ~ normal(xy[i-1,:]+w_xy[i-1,:], q1);
        w_xy[i,:] ~ normal(w_xy[i-1,:], q2);

        // measurement likelihood
        y_xy[i,:] ~ normal(xy[i,:], r);
    }
//    omega[2:N] ~ normal(omega[1:N-1], q);
//    theta[2:N] ~ normal(theta[1:N-1] + omega[1:N-1], q);

    // measurement likelihood


}
generated quantities {
vector<lower = -pi(), upper = pi()>[N] theta;
vector[N] omega;
    for (i in 1:N){
        theta[i] = atan2(xy[i,2], xy[i,1]);
        omega[i] = atan2(w_xy[i,2], w_xy[i,1]);
    }
}