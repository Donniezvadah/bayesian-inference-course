# Bayesian design for a Beta-Binomial experiment with linear sampling cost.

set.seed(20260327)

expected_posterior_variance <- function(n, alpha, beta) {
  alpha * beta / ((alpha + beta) * (alpha + beta + 1) * (alpha + beta + n))
}

expected_utility_beta_binomial <- function(n, alpha, beta, cost) {
  -expected_posterior_variance(n, alpha, beta) - cost * n
}

mc_expected_utility <- function(n, alpha, beta, cost, S = 5000) {
  theta <- rbeta(S, alpha, beta)
  y <- rbinom(S, size = n, prob = theta)
  post_var <- (alpha + y) * (beta + n - y) /
    ((alpha + beta + n)^2 * (alpha + beta + n + 1))
  mean(-post_var - cost * n)
}

alpha <- 2
beta <- 3
cost <- 0.0015
grid_n <- 0:120

analytic_utility <- sapply(grid_n, expected_utility_beta_binomial, alpha = alpha, beta = beta, cost = cost)
mc_utility <- sapply(grid_n, mc_expected_utility, alpha = alpha, beta = beta, cost = cost, S = 2000)

best_n_analytic <- grid_n[which.max(analytic_utility)]
best_n_mc <- grid_n[which.max(mc_utility)]

summary_table <- data.frame(
  n = grid_n[seq(1, length(grid_n), by = 10)],
  analytic = round(analytic_utility[seq(1, length(grid_n), by = 10)], 5),
  monte_carlo = round(mc_utility[seq(1, length(grid_n), by = 10)], 5)
)

cat("Best analytic design n:\n")
print(best_n_analytic)
cat("\nBest Monte Carlo design n:\n")
print(best_n_mc)
cat("\nSample of expected utilities across candidate designs\n")
print(summary_table, row.names = FALSE)
