data {
  int<lower=1> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
}

parameters {
  real mu;
  real<lower=0> tau;
  vector[J] z;
}

transformed parameters {
  vector[J] theta;
  theta = mu + tau * z;
}

model {
  mu ~ normal(0, 5);
  tau ~ normal(0, 5);
  z ~ normal(0, 1);
  y ~ normal(theta, sigma);
}

generated quantities {
  array[J] real y_rep;
  vector[J] log_lik;

  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma[j]);
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
  }
}
