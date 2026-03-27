# Bootstrap particle filter for a local-level state-space model.

set.seed(2026)

simulate_local_level <- function(T, W, V, x0 = 0) {
  x <- numeric(T)
  y <- numeric(T)
  x[1] <- x0 + rnorm(1, 0, sqrt(W))
  y[1] <- x[1] + rnorm(1, 0, sqrt(V))

  for (t in 2:T) {
    x[t] <- x[t - 1] + rnorm(1, 0, sqrt(W))
    y[t] <- x[t] + rnorm(1, 0, sqrt(V))
  }

  list(x = x, y = y)
}

bootstrap_particle_filter <- function(y, N, W, V, resample_threshold = N / 2) {
  T <- length(y)
  particles <- rnorm(N, 0, sqrt(W))
  weights <- rep(1 / N, N)
  filtered_mean <- numeric(T)
  ess_history <- numeric(T)

  for (t in seq_len(T)) {
    if (t > 1) {
      particles <- particles + rnorm(N, 0, sqrt(W))
    }

    weights <- dnorm(y[t], mean = particles, sd = sqrt(V))
    weights <- weights / sum(weights)

    filtered_mean[t] <- sum(weights * particles)
    ess_history[t] <- 1 / sum(weights^2)

    if (ess_history[t] < resample_threshold) {
      ancestors <- sample.int(N, size = N, replace = TRUE, prob = weights)
      particles <- particles[ancestors]
      weights <- rep(1 / N, N)
    }
  }

  list(filtered_mean = filtered_mean, ess = ess_history)
}

sim <- simulate_local_level(T = 80, W = 0.4, V = 1.0)
pf <- bootstrap_particle_filter(sim$y, N = 1000, W = 0.4, V = 1.0)

par(mfrow = c(2, 1))
plot(sim$x, type = "l", lwd = 2, col = "black", ylim = range(c(sim$x, pf$filtered_mean)),
     main = "True state vs particle filter mean", xlab = "Time", ylab = "State")
lines(pf$filtered_mean, col = "firebrick", lwd = 2)
legend("topleft", legend = c("True state", "Filtered mean"),
       col = c("black", "firebrick"), lwd = 2, bty = "n")

plot(pf$ess, type = "l", lwd = 2, col = "steelblue",
     main = "Particle ESS over time", xlab = "Time", ylab = "ESS")
