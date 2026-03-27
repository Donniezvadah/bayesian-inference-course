# Random-walk Metropolis for a conjugate normal posterior

log_post_theta <- function(theta, y, sigma, mu0, tau0) {
  loglik <- sum(dnorm(y, mean = theta, sd = sigma, log = TRUE))
  logprior <- dnorm(theta, mean = mu0, sd = tau0, log = TRUE)
  loglik + logprior
}

rw_metropolis <- function(init, n_iter, proposal_sd, y, sigma, mu0, tau0) {
  draws <- numeric(n_iter)
  draws[1] <- init
  accept <- 0L

  for (t in 2:n_iter) {
    proposal <- rnorm(1, mean = draws[t - 1], sd = proposal_sd)
    log_alpha <- log_post_theta(proposal, y, sigma, mu0, tau0) -
      log_post_theta(draws[t - 1], y, sigma, mu0, tau0)

    if (log(runif(1)) < log_alpha) {
      draws[t] <- proposal
      accept <- accept + 1L
    } else {
      draws[t] <- draws[t - 1]
    }
  }

  list(draws = draws, accept_rate = accept / (n_iter - 1))
}

effective_sample_size <- function(x, max_lag = 100) {
  acf_vals <- acf(x, plot = FALSE, lag.max = max_lag)$acf[-1]
  positive_acf <- acf_vals[acf_vals > 0]
  n <- length(x)
  n / (1 + 2 * sum(positive_acf))
}

set.seed(42)
y <- rnorm(40, mean = 1.5, sd = 2)
sigma <- 2
mu0 <- 0
tau0 <- 5

n <- length(y)
post_var <- 1 / (n / sigma^2 + 1 / tau0^2)
post_mean <- post_var * (n * mean(y) / sigma^2 + mu0 / tau0^2)

fit <- rw_metropolis(
  init = 0,
  n_iter = 12000,
  proposal_sd = 0.35,
  y = y,
  sigma = sigma,
  mu0 = mu0,
  tau0 = tau0
)

burn_in <- 2000
draws <- fit$draws[(burn_in + 1):length(fit$draws)]

cat("Acceptance rate:", fit$accept_rate, "\n")
cat("Posterior mean estimate:", mean(draws), "\n")
cat("Exact posterior mean:", post_mean, "\n")
cat("Approximate ESS:", effective_sample_size(draws), "\n")

par(mfrow = c(1, 2))
plot(draws, type = "l", col = "gray40", main = "Trace plot", ylab = expression(theta))
hist(draws, breaks = 40, freq = FALSE, col = "gray85",
     main = "Posterior draws", xlab = expression(theta))
curve(dnorm(x, mean = post_mean, sd = sqrt(post_var)),
      add = TRUE, col = "firebrick", lwd = 2)
