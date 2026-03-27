# Bayes estimators and classification thresholds from posterior draws

set.seed(321)

posterior_draws <- rbeta(50000, shape1 = 16, shape2 = 8)

bayes_squared <- mean(posterior_draws)
bayes_absolute <- median(posterior_draws)
bayes_mode <- (16 - 1) / (16 + 8 - 2)

cat("Posterior mean (quadratic loss):", bayes_squared, "\n")
cat("Posterior median (absolute loss):", bayes_absolute, "\n")
cat("Posterior mode (0-1 style modal loss):", bayes_mode, "\n")

false_positive_cost <- 1
false_negative_cost <- 4
threshold <- false_positive_cost / (false_positive_cost + false_negative_cost)

cat("Bayes classification threshold:", threshold, "\n")

grid <- seq(0.01, 0.99, length.out = 300)
risk_squared <- sapply(grid, function(a) mean((a - posterior_draws)^2))
risk_absolute <- sapply(grid, function(a) mean(abs(a - posterior_draws)))

par(mfrow = c(1, 2))
plot(grid, risk_squared, type = "l", lwd = 2, col = "steelblue",
     xlab = "Action", ylab = "Posterior risk", main = "Quadratic loss")
abline(v = bayes_squared, col = "firebrick", lwd = 2)

plot(grid, risk_absolute, type = "l", lwd = 2, col = "steelblue",
     xlab = "Action", ylab = "Posterior risk", main = "Absolute loss")
abline(v = bayes_absolute, col = "firebrick", lwd = 2)
