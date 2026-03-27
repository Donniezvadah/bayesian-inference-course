# Simple simulation-based calibration for a normal mean model.

set.seed(2026)
n_sims <- 200
n_post <- 500
n <- 25
sigma <- 1

ranks <- integer(n_sims)

for (s in seq_len(n_sims)) {
  theta_true <- rnorm(1, 0, 1)
  y <- rnorm(n, theta_true, sigma)

  post_var <- 1 / (n / sigma^2 + 1)
  post_mean <- post_var * sum(y) / sigma^2
  post_draws <- rnorm(n_post, post_mean, sqrt(post_var))

  ranks[s] <- sum(post_draws < theta_true)
}

hist(ranks, breaks = 20, col = "gray85",
     main = "SBC rank histogram", xlab = "Rank of true parameter")
