# Compare Gaussian and Student-t MAP regression under outliers.

set.seed(20260327)

neg_log_post_gaussian <- function(beta, X, y, sigma, tau) {
  resid <- y - drop(X %*% beta)
  ll <- -0.5 * sum(resid^2) / sigma^2
  lp <- -0.5 * sum(beta^2) / tau^2
  -(ll + lp)
}

neg_log_post_t <- function(beta, X, y, sigma, tau, nu) {
  resid <- y - drop(X %*% beta)
  ll <- -0.5 * (nu + 1) * sum(log1p(resid^2 / (nu * sigma^2)))
  lp <- -0.5 * sum(beta^2) / tau^2
  -(ll + lp)
}

n <- 120
x <- seq(-2, 2, length.out = n)
X <- cbind(1, x)
beta_true <- c(1.0, 1.8)
sigma <- 0.4
tau <- 5
nu <- 4

y <- drop(X %*% beta_true + rnorm(n, sd = sigma))
outlier_idx <- c(18, 64, 97, 112)
y[outlier_idx] <- y[outlier_idx] + c(4.0, -5.5, 4.8, -4.2)

fit_gaussian <- optim(
  par = c(0, 0),
  fn = neg_log_post_gaussian,
  X = X,
  y = y,
  sigma = sigma,
  tau = tau,
  method = "BFGS"
)

fit_t <- optim(
  par = c(0, 0),
  fn = neg_log_post_t,
  X = X,
  y = y,
  sigma = sigma,
  tau = tau,
  nu = nu,
  method = "BFGS"
)

beta_gaussian <- fit_gaussian$par
beta_t <- fit_t$par

resid_t <- y - drop(X %*% beta_t)
weights_t <- (nu + 1) / (nu + (resid_t / sigma)^2)

cat("True coefficients\n")
print(beta_true)
cat("\nGaussian MAP coefficients\n")
print(round(beta_gaussian, 4))
cat("\nStudent-t MAP coefficients\n")
print(round(beta_t, 4))
cat("\nSmallest implied Student-t weights\n")
print(round(sort(weights_t)[1:8], 4))
cat("\nOutlier indices and their Student-t weights\n")
print(round(weights_t[outlier_idx], 4))
