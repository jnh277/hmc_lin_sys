data {
    int<lower=0> N;
    vector[N] y;
}
parameters {
    real<lower=-1.0,upper=1.0> a;
    real<lower=0.0> r;  // measurement noise std
    real<lower=0.0> q;  // process noise std
    vector[N-1] w;        // process model errors
    real z0;            // initial state
}
transformed parameters{
    vector[N] z;
//    matrix[N,N] w2z;
//    vector[N] z02z;
//    w2z = rep_matrix(0,N,N);
//    for (i in 1:N){
//        z02z[i] = pow(a,i);
//        for(j in 1:i){
//            w2z[i,j] = pow(a,i-j);
//        }
//    }
//    print(z02z);
//    print(w2z);
//    z = z02z * z0 + w2z*w;
    z[1] = z0;
    for (i in 1:N-1){
        z[i+1] = a*z[i] + w[i];
    }
}
model {
    // noise stds priors
    r ~ cauchy(0,1.0);
    q ~ cauchy(0, 1.0);

    // prior on parameter
    a ~ cauchy(0, 1.0);

    // process noise likelihood
    w ~ normal(0,q);        // process noise likelihood

    // measurement likelihood
    y ~ normal(z, r);

}