# Prior predictive simulation for Bayesian logistic regression

simulate_prior_predictive <- function(n_draws, x, beta0_sd, beta1_sd) {
  beta0 <- rnorm(n_draws, mean = 0, sd = beta0_sd)
  beta1 <- rnorm(n_draws, mean = 0, sd = beta1_sd)

  probs <- sapply(seq_len(n_draws), function(i) {
    plogis(beta0[i] + beta1[i] * x)
  })

  list(beta0 = beta0, beta1 = beta1, probs = probs)
}

set.seed(2026)
x <- seq(-2, 2, length.out = 100)

weak <- simulate_prior_predictive(200, x, beta0_sd = 1.5, beta1_sd = 1.0)
diffuse <- simulate_prior_predictive(200, x, beta0_sd = 5, beta1_sd = 5)

par(mfrow = c(1, 2))
matplot(
  x, weak$probs, type = "l", lty = 1, col = rgb(0, 0, 0, 0.05),
  xlab = "x", ylab = "Success probability", main = "Weakly informative prior"
)
matplot(
  x, diffuse$probs, type = "l", lty = 1, col = rgb(0, 0, 0, 0.05),
  xlab = "x", ylab = "Success probability", main = "Diffuse prior"
)
