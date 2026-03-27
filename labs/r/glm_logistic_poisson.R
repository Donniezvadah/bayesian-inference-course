# Bayesian-style posterior functions for logistic and Poisson regression.

log_post_logistic <- function(beta, X, y, prior_sd = 2.5) {
  eta <- drop(X %*% beta)
  loglik <- sum(y * eta - log1p(exp(eta)))
  logprior <- sum(dnorm(beta, mean = 0, sd = prior_sd, log = TRUE))
  loglik + logprior
}

grad_log_post_logistic <- function(beta, X, y, prior_sd = 2.5) {
  eta <- drop(X %*% beta)
  p <- plogis(eta)
  drop(crossprod(X, y - p) - beta / prior_sd^2)
}

log_post_poisson <- function(beta, X, y, prior_sd = 2.5) {
  eta <- drop(X %*% beta)
  loglik <- sum(y * eta - exp(eta) - lfactorial(y))
  logprior <- sum(dnorm(beta, mean = 0, sd = prior_sd, log = TRUE))
  loglik + logprior
}

set.seed(2026)
n <- 200
x <- rnorm(n)
X <- cbind(1, x)

beta_true_logit <- c(-0.3, 1.1)
p <- plogis(drop(X %*% beta_true_logit))
y_bin <- rbinom(n, 1, p)

beta0 <- c(0, 0)
cat("Log posterior at beta0:", log_post_logistic(beta0, X, y_bin), "\n")
cat("Gradient at beta0:\n")
print(grad_log_post_logistic(beta0, X, y_bin))

beta_true_pois <- c(0.4, 0.5)
lambda <- exp(drop(X %*% beta_true_pois))
y_count <- rpois(n, lambda)
cat("Poisson log posterior at beta0:", log_post_poisson(beta0, X, y_count), "\n")
