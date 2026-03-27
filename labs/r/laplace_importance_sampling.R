# Compare exact, Laplace, and importance-sampling summaries
# for a Gamma posterior from a Poisson-Gamma model.

log_post_lambda <- function(lambda, y, alpha, beta) {
  ifelse(
    lambda > 0,
    (alpha + sum(y) - 1) * log(lambda) - (beta + length(y)) * lambda,
    -Inf
  )
}

laplace_gamma_posterior <- function(y, alpha, beta) {
  shape_post <- alpha + sum(y)
  rate_post <- beta + length(y)
  mode <- (shape_post - 1) / rate_post
  precision <- (shape_post - 1) / mode^2

  list(
    mode = mode,
    var = 1 / precision,
    mean_exact = shape_post / rate_post,
    var_exact = shape_post / rate_post^2
  )
}

importance_sampling_mean <- function(n_draws, proposal_mean, proposal_sd, y, alpha, beta) {
  draws <- rnorm(n_draws, mean = proposal_mean, sd = proposal_sd)
  draws <- draws[draws > 0]

  log_weights <- log_post_lambda(draws, y, alpha, beta) -
    dnorm(draws, mean = proposal_mean, sd = proposal_sd, log = TRUE)
  weights <- exp(log_weights - max(log_weights))

  list(
    estimate = sum(weights * draws) / sum(weights),
    ess = (sum(weights)^2) / sum(weights^2),
    draws = draws,
    weights = weights
  )
}

set.seed(2026)
y <- c(3, 2, 4, 1, 0, 3, 5, 2)
alpha <- 2
beta <- 1

laplace_fit <- laplace_gamma_posterior(y, alpha, beta)
is_fit <- importance_sampling_mean(
  n_draws = 20000,
  proposal_mean = laplace_fit$mode,
  proposal_sd = sqrt(laplace_fit$var),
  y = y,
  alpha = alpha,
  beta = beta
)

cat("Exact posterior mean:", laplace_fit$mean_exact, "\n")
cat("Laplace mode:", laplace_fit$mode, "\n")
cat("Laplace variance:", laplace_fit$var, "\n")
cat("Importance sampling estimate:", is_fit$estimate, "\n")
cat("Importance sampling ESS:", is_fit$ess, "\n")

grid <- seq(0.001, 8, length.out = 600)
shape_post <- alpha + sum(y)
rate_post <- beta + length(y)
exact_density <- dgamma(grid, shape = shape_post, rate = rate_post)
laplace_density <- dnorm(grid, mean = laplace_fit$mode, sd = sqrt(laplace_fit$var))

plot(grid, exact_density, type = "l", lwd = 2, col = "firebrick",
     xlab = expression(lambda), ylab = "Density",
     main = "Exact posterior vs Laplace approximation")
lines(grid, laplace_density, lwd = 2, col = "steelblue")
legend("topright", legend = c("Exact Gamma posterior", "Laplace Gaussian"),
       col = c("firebrick", "steelblue"), lwd = 2, bty = "n")
