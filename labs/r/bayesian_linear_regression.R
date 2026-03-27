# Bayesian linear regression with known variance and Gaussian prior.

posterior_linear_known_sigma <- function(X, y, sigma2, b0, B0) {
  Bn <- solve(crossprod(X) / sigma2 + solve(B0))
  bn <- Bn %*% (crossprod(X, y) / sigma2 + solve(B0, b0))
  list(mean = drop(bn), cov = Bn)
}

posterior_predictive_draws <- function(X_new, post_mean, post_cov, sigma2, n_draws = 4000) {
  L <- chol(post_cov)
  z <- matrix(rnorm(length(post_mean) * n_draws), ncol = length(post_mean))
  beta_draws <- sweep(z %*% L, 2, post_mean, `+`)
  mu_draws <- beta_draws %*% t(X_new)
  mu_draws + matrix(rnorm(nrow(mu_draws) * ncol(mu_draws), 0, sqrt(sigma2)),
                    nrow = nrow(mu_draws))
}

set.seed(2026)
n <- 120
x <- seq(-2, 2, length.out = n)
X <- cbind(1, x)
beta_true <- c(1.0, 1.8)
sigma2 <- 1.0
y <- drop(X %*% beta_true + rnorm(n, 0, sqrt(sigma2)))

b0 <- c(0, 0)
B0 <- diag(c(25, 4))
post <- posterior_linear_known_sigma(X, y, sigma2, b0, B0)

cat("Posterior mean:\n")
print(post$mean)
cat("Posterior covariance:\n")
print(post$cov)

X_new <- cbind(1, seq(-2.5, 2.5, length.out = 80))
pred <- posterior_predictive_draws(X_new, post$mean, post$cov, sigma2)
pred_mean <- colMeans(pred)
pred_low <- apply(pred, 2, quantile, 0.05)
pred_high <- apply(pred, 2, quantile, 0.95)

plot(x, y, pch = 16, col = rgb(0, 0, 0, 0.35),
     xlab = "x", ylab = "y", main = "Bayesian linear regression")
lines(X_new[, 2], pred_mean, col = "firebrick", lwd = 2)
lines(X_new[, 2], pred_low, col = "steelblue", lwd = 2, lty = 2)
lines(X_new[, 2], pred_high, col = "steelblue", lwd = 2, lty = 2)
