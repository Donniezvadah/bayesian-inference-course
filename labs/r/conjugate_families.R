# Exact updates for core conjugate Bayesian models

beta_binomial_update <- function(y, n, alpha, beta) {
  c(
    alpha_post = alpha + y,
    beta_post = beta + n - y
  )
}

beta_binomial_predictive <- function(k, m, y, n, alpha, beta) {
  choose(m, k) *
    beta(alpha + y + k, beta + n - y + m - k) /
    beta(alpha + y, beta + n - y)
}

gamma_poisson_update <- function(y, alpha, beta) {
  c(
    shape_post = alpha + sum(y),
    rate_post = beta + length(y)
  )
}

normal_normal_update <- function(y, sigma2, mu0, tau02) {
  n <- length(y)
  tau_n2 <- 1 / (n / sigma2 + 1 / tau02)
  mu_n <- tau_n2 * (n * mean(y) / sigma2 + mu0 / tau02)
  c(mu_post = mu_n, tau2_post = tau_n2)
}

dirichlet_multinomial_update <- function(y, alpha) {
  alpha + y
}

set.seed(101)

cat("Beta-Binomial:\n")
print(beta_binomial_update(y = 14, n = 20, alpha = 2, beta = 2))

cat("\nGamma-Poisson:\n")
print(gamma_poisson_update(y = c(2, 1, 4, 0, 3), alpha = 3, beta = 2))

cat("\nNormal-Normal:\n")
print(normal_normal_update(y = rnorm(12, 1.5, 2), sigma2 = 4, mu0 = 0, tau02 = 25))

cat("\nDirichlet-Multinomial:\n")
print(dirichlet_multinomial_update(y = c(12, 8, 5), alpha = c(1, 1, 1)))
