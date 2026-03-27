# Proposal-scale tuning grid for random-walk Metropolis.

source("labs/r/mh_sampler.R")

set.seed(2026)
y <- rnorm(40, mean = 1.5, sd = 2)
sigma <- 2
mu0 <- 0
tau0 <- 5

proposal_grid <- c(0.05, 0.15, 0.35, 0.7, 1.4)
results <- data.frame(
  proposal_sd = proposal_grid,
  accept_rate = NA_real_,
  ess = NA_real_
)

for (i in seq_along(proposal_grid)) {
  fit <- rw_metropolis(
    init = 0,
    n_iter = 12000,
    proposal_sd = proposal_grid[i],
    y = y,
    sigma = sigma,
    mu0 = mu0,
    tau0 = tau0
  )
  draws <- fit$draws[2001:12000]
  results$accept_rate[i] <- fit$accept_rate
  results$ess[i] <- effective_sample_size(draws)
}

print(results)
