# Normal-normal partial pooling illustration.

set.seed(2026)
J <- 12
mu <- 0.8
tau2 <- 1.2

sigma2_j <- runif(J, 0.2, 2.0)
theta_true <- rnorm(J, mu, sqrt(tau2))
ybar <- rnorm(J, theta_true, sqrt(sigma2_j))

post_var <- 1 / (1 / sigma2_j + 1 / tau2)
post_mean <- post_var * (ybar / sigma2_j + mu / tau2)

pooling_weight <- tau2 / (tau2 + sigma2_j)

cat("Posterior means:\n")
print(round(post_mean, 3))
cat("Pooling weights:\n")
print(round(pooling_weight, 3))

plot(ybar, post_mean, pch = 16, col = "firebrick",
     xlab = "No-pooling estimate", ylab = "Partial-pooling estimate",
     main = "Hierarchical shrinkage")
abline(0, 1, col = "gray50", lty = 2)
abline(h = mu, col = "steelblue", lwd = 2)
