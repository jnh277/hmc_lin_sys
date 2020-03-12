data {
    int<lower=0> N;
    vector[N] y;
}
parameters {
    real<lower=-1.0,upper=1.0> a;
    real<lower=0.0> r;
    real<lower=0.0> q;
    vector[N] z;
}
model {
    // noise stds priors
    r ~ cauchy(0,1.0);
    q ~ cauchy(0, 1.0);

    // prior on parameter
    a ~ cauchy(0, 1.0);

    // initial state prior
    z[1] ~ cauchy(0,5);

    // state likelihood
    z[2:N] ~ normal(a*z[1:N-1], q);

    // measurement likelihood
    y ~ normal(z, r);

}