data {
    int<lower=0> N;
    vector[N] y;
    real muP;      // initial state prior mean
    real cP;        // initial state prior variance
}
parameters {
    real a;
    real<lower=-1.0,upper=1.0> b;
    real<lower=1e-8> c;
    vector[N] z;                    // state = log volatility
}
model {
    // parameter priors
//    c ~ cauchy(0, 5.0);
//
//    // prior on parameters
//    a ~ cauchy(0, 10.0);
//    b ~ uniform(-1, 1);

    // initial state prior
    z[1] ~ normal(muP,cP);

    // state likelihood
    z[2:N] ~ normal(a+b*z[1:N-1], sqrt(c));

    // measurement likelihood
    y ~ normal(0, sqrt(exp(z)));

}

generated quantities {
real lpd = 0;
lpd += normal_lpdf(y | 0, sqrt(exp(z)));
lpd += normal_lpdf(z[2:N] | a+b*z[1:N-1], sqrt(c));
//c = c_std * c_std;
}