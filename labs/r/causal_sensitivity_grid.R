# Simple causal sensitivity grid for a treatment-effect regression.

set.seed(2026)
n <- 250
x <- rnorm(n)
a <- rbinom(n, 1, plogis(0.3 * x))
y <- 1.5 * a + 0.8 * x + rnorm(n)

fit_naive <- lm(y ~ a + x)
naive_effect <- coef(fit_naive)[["a"]]

sensitivity_grid <- seq(-1, 1, length.out = 41)
adjusted_effect <- naive_effect - sensitivity_grid

plot(sensitivity_grid, adjusted_effect, type = "l", lwd = 2, col = "firebrick",
     xlab = expression(delta), ylab = "Implied treatment effect",
     main = "Sensitivity of treatment effect to hidden bias parameter")
abline(h = 0, lty = 2, col = "gray50")
