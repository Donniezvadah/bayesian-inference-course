# Likelihood geometry examples:
# Binomial MLE, Poisson curvature, profile likelihood, and posterior comparison.

set.seed(20260327)

loglik_binomial <- function(theta, y, n) {
  ifelse(theta > 0 & theta < 1,
         y * log(theta) + (n - y) * log(1 - theta),
         -Inf)
}

profile_loglik_mu <- function(mu, y) {
  sigma2_hat <- mean((y - mu)^2)
  -0.5 * length(y) * log(sigma2_hat)
}

# Binomial example
y_bin <- 18
n_bin <- 30
theta_grid <- seq(0.01, 0.99, length.out = 500)
loglik_vals <- loglik_binomial(theta_grid, y = y_bin, n = n_bin)
theta_mle <- theta_grid[which.max(loglik_vals)]

# Weakly informative Bayesian comparison
alpha <- 2
beta <- 2
posterior_grid <- exp(loglik_vals + dbeta(theta_grid, alpha, beta, log = TRUE))
posterior_grid <- posterior_grid / sum(posterior_grid)
posterior_mean <- sum(theta_grid * posterior_grid)
posterior_cdf <- cumsum(posterior_grid)
cdf_unique <- !duplicated(posterior_cdf)
credible_interval <- approx(
  x = posterior_cdf[cdf_unique],
  y = theta_grid[cdf_unique],
  xout = c(0.025, 0.975)
)$y

# Poisson curvature
y_pois <- c(3, 4, 2, 6, 5, 3, 4, 5)
lambda_hat <- mean(y_pois)
observed_information <- sum(y_pois) / lambda_hat^2

# Profile likelihood for normal mean with unknown variance
y_norm <- c(1.3, 0.7, 1.8, 2.1, 0.5, 1.6, 1.0, 1.4, 2.2, 0.9)
mu_grid <- seq(-0.5, 3, length.out = 400)
profile_vals <- sapply(mu_grid, profile_loglik_mu, y = y_norm)
mu_profile_hat <- mu_grid[which.max(profile_vals)]
cutoff <- max(profile_vals) - 0.5 * qchisq(0.95, df = 1)
profile_interval <- range(mu_grid[profile_vals >= cutoff])

cat("Binomial MLE for theta\n")
print(round(theta_mle, 4))
cat("\nPosterior mean under Beta(2,2) prior\n")
print(round(posterior_mean, 4))
cat("\nApproximate 95% equal-tail credible interval\n")
print(round(credible_interval, 4))

cat("\nPoisson MLE and observed information\n")
print(round(c(lambda_hat = lambda_hat, observed_information = observed_information), 4))

cat("\nProfile likelihood estimate and approximate 95% profile interval for mu\n")
print(round(c(mu_hat = mu_profile_hat, lower = profile_interval[1], upper = profile_interval[2]), 4))
