# Exact Gaussian process regression in one dimension.

rbf_kernel <- function(x1, x2, ell = 0.8, alpha = 1.0) {
  outer(x1, x2, function(a, b) alpha^2 * exp(-0.5 * (a - b)^2 / ell^2))
}

set.seed(2026)
x <- seq(-2, 2, length.out = 30)
f_true <- sin(1.5 * x)
y <- f_true + rnorm(length(x), 0, 0.2)

x_star <- seq(-2.5, 2.5, length.out = 200)
K <- rbf_kernel(x, x) + 0.2^2 * diag(length(x))
Ks <- rbf_kernel(x_star, x)
Kss <- rbf_kernel(x_star, x_star)

K_inv <- solve(K)
post_mean <- Ks %*% K_inv %*% y
post_cov <- Kss - Ks %*% K_inv %*% t(Ks)
post_sd <- sqrt(pmax(diag(post_cov), 0))

plot(x, y, pch = 16, col = "black",
     main = "Gaussian process regression", xlab = "x", ylab = "y")
lines(x_star, post_mean, col = "firebrick", lwd = 2)
lines(x_star, post_mean + 2 * post_sd, col = "steelblue", lty = 2, lwd = 2)
lines(x_star, post_mean - 2 * post_sd, col = "steelblue", lty = 2, lwd = 2)
