# Fit the non-centered eight-schools model with cmdstanr

library(cmdstanr)
library(posterior)
library(bayesplot)
library(loo)

schools_data <- list(
  J = 8,
  y = c(28, 8, -3, 7, -1, 1, 18, 12),
  sigma = c(15, 10, 16, 11, 9, 11, 10, 18)
)

mod <- cmdstan_model("labs/stan/eight_schools.stan")

fit <- mod$sample(
  data = schools_data,
  seed = 1234,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  adapt_delta = 0.95,
  max_treedepth = 12
)

print(fit$summary(c("mu", "tau", "theta")))
print(fit$cmdstan_diagnose())

draws_yrep <- fit$draws("y_rep", format = "draws_matrix")
bayesplot::ppc_dens_overlay(
  y = schools_data$y,
  yrep = draws_yrep[1:200, ]
)

log_lik_draws <- fit$draws("log_lik", format = "matrix")
print(loo(log_lik_draws))
