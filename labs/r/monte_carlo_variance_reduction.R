# Monte Carlo integration, variance reduction, and weighted ESS.

set.seed(20260327)

f <- function(u) exp(u)
exact_integral <- exp(1) - 1

S <- 5000
u <- runif(S)
x <- f(u)

crude_estimate <- mean(x)
crude_mcse <- sd(x) / sqrt(S)

# Antithetic estimator using pairs.
u_half <- runif(S / 2)
anti_values <- 0.5 * (f(u_half) + f(1 - u_half))
antithetic_estimate <- mean(anti_values)
antithetic_mcse <- sd(anti_values) / sqrt(length(anti_values))

# Control variate using U with known mean 1/2.
a_star <- cov(x, u) / var(u)
cv_values <- x - a_star * (u - 0.5)
cv_estimate <- mean(cv_values)
cv_mcse <- sd(cv_values) / sqrt(S)

# Importance-sampling example for E_p[theta^2], p = N(0,1), q = N(0, 2^2).
theta <- rnorm(S, mean = 0, sd = 2)
weights <- dnorm(theta, mean = 0, sd = 1) / dnorm(theta, mean = 0, sd = 2)
weights_norm <- weights / sum(weights)
is_estimate <- sum(weights_norm * theta^2)
ess <- (sum(weights)^2) / sum(weights^2)

summary_table <- data.frame(
  estimator = c("crude", "antithetic", "control_variate"),
  estimate = round(c(crude_estimate, antithetic_estimate, cv_estimate), 6),
  mcse = round(c(crude_mcse, antithetic_mcse, cv_mcse), 6)
)

cat("Exact integral E[exp(U)] with U ~ Unif(0,1)\n")
print(round(exact_integral, 6))

cat("\nMonte Carlo estimators and MCSEs\n")
print(summary_table, row.names = FALSE)

cat("\nImportance-sampling estimate of E_p[theta^2] under p = N(0,1)\n")
print(round(c(
  estimate = is_estimate,
  exact = 1,
  ess = ess
), 4))
