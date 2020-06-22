
data {
    int<lower=1> N;
    vector[N] y;
    real<lower=0> sig_e;
}
parameters {
    real<lower=-1,upper=1> a;
}
model {
    a ~ normal(0, 5);
    y[2:N] ~ normal(a*y[1:N-1],sig_e);

}