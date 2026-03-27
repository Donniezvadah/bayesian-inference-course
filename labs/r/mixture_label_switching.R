# Simple Gibbs-style allocation updates for a two-component Gaussian mixture.

set.seed(2026)
n <- 200
mu_true <- c(-2, 2)
z_true <- rbinom(n, 1, 0.5) + 1
y <- rnorm(n, mean = mu_true[z_true], sd = 1)

mu <- c(-1, 1)
pi_k <- c(0.5, 0.5)

allocation_probs <- function(y_i, mu, pi_k) {
  numer <- pi_k * dnorm(y_i, mean = mu, sd = 1)
  numer / sum(numer)
}

z <- integer(n)
for (i in seq_len(n)) {
  z[i] <- sample(1:2, size = 1, prob = allocation_probs(y[i], mu, pi_k))
}

table(z)

plot(density(y), lwd = 2, main = "Mixture data and initial component means")
abline(v = mu, col = c("firebrick", "steelblue"), lwd = 2)
