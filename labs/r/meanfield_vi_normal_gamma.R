# Mean-field coordinate ascent variational inference for a Normal-Gamma model.

set.seed(2026)
y <- rnorm(80, mean = 1.3, sd = 1.8)

mu0 <- 0
kappa0 <- 1
a0 <- 2
b0 <- 2

n_iter <- 200

a <- a0 + (length(y) + 1) / 2
b <- b0 + 1

m_history <- numeric(n_iter)
s2_history <- numeric(n_iter)
a_history <- numeric(n_iter)
b_history <- numeric(n_iter)

for (iter in seq_len(n_iter)) {
  expected_tau <- a / b

  m <- (kappa0 * mu0 + sum(y)) / (kappa0 + length(y))
  s2 <- 1 / ((kappa0 + length(y)) * expected_tau)

  quad_y <- sum((y - m)^2) + length(y) * s2
  quad_mu <- (m - mu0)^2 + s2

  a <- a0 + (length(y) + 1) / 2
  b <- b0 + 0.5 * (quad_y + kappa0 * quad_mu)

  m_history[iter] <- m
  s2_history[iter] <- s2
  a_history[iter] <- a
  b_history[iter] <- b
}

cat("Variational mean for mu:", tail(m_history, 1), "\n")
cat("Variational variance for mu:", tail(s2_history, 1), "\n")
cat("Variational E[tau]:", tail(a_history / b_history, 1), "\n")

par(mfrow = c(1, 2))
plot(m_history, type = "l", lwd = 2, col = "steelblue",
     main = "CAVI mean update", xlab = "Iteration", ylab = "m")
plot(s2_history, type = "l", lwd = 2, col = "firebrick",
     main = "CAVI variance update", xlab = "Iteration", ylab = expression(s^2))
