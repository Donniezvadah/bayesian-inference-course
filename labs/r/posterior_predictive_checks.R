# Posterior predictive check for a simple normal model.

set.seed(2026)
y <- rnorm(80, mean = 0.8, sd = 1.2)

mu_post <- rnorm(4000, mean = mean(y), sd = sd(y) / sqrt(length(y)))
sigma_post <- abs(rnorm(4000, mean = sd(y), sd = 0.12))

y_rep_mean <- numeric(length(mu_post))
y_rep_var <- numeric(length(mu_post))

for (i in seq_along(mu_post)) {
  y_rep <- rnorm(length(y), mu_post[i], sigma_post[i])
  y_rep_mean[i] <- mean(y_rep)
  y_rep_var[i] <- var(y_rep)
}

par(mfrow = c(1, 2))
hist(y_rep_mean, breaks = 40, col = "gray85",
     main = "PPC for mean", xlab = "Replicated mean")
abline(v = mean(y), col = "firebrick", lwd = 2)

hist(y_rep_var, breaks = 40, col = "gray85",
     main = "PPC for variance", xlab = "Replicated variance")
abline(v = var(y), col = "firebrick", lwd = 2)
