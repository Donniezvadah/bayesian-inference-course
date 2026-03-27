# Posterior reporting on transformed scales and predictive uncertainty.

set.seed(20260327)

summarize_draws <- function(x) {
  c(
    mean = mean(x),
    sd = sd(x),
    q05 = unname(quantile(x, 0.05)),
    q50 = unname(quantile(x, 0.50)),
    q95 = unname(quantile(x, 0.95))
  )
}

S <- 4000
beta0 <- rnorm(S, mean = -0.3, sd = 0.25)
beta1 <- rnorm(S, mean = 0.9, sd = 0.18)
sigma <- abs(rnorm(S, mean = 0.6, sd = 0.07))

x_control <- c(1, 0.0)
x_treated <- c(1, 1.0)

inv_logit <- function(z) 1 / (1 + exp(-z))

p_control <- inv_logit(beta0 + beta1 * x_control[2])
p_treated <- inv_logit(beta0 + beta1 * x_treated[2])
risk_difference <- p_treated - p_control
odds_ratio <- exp(beta1)

mu_pred <- beta0 + beta1 * 1.5
y_pred <- rnorm(S, mean = mu_pred, sd = sigma)

noise_component <- mean(sigma^2)
parameter_component <- var(mu_pred)
predictive_variance <- var(y_pred)

cat("Posterior summary: odds ratio\n")
print(round(summarize_draws(odds_ratio), 4))
cat("\nPosterior summary: risk difference at x = 0 versus x = 1\n")
print(round(summarize_draws(risk_difference), 4))
cat("\nPosterior probability risk difference > 0.10\n")
print(round(mean(risk_difference > 0.10), 4))
cat("\nPredictive variance decomposition at x = 1.5\n")
print(round(c(
  expected_noise = noise_component,
  parameter_uncertainty = parameter_component,
  predictive_variance = predictive_variance
), 4))
