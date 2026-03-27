# Forward algorithm for a simple 2-state hidden Markov model.

transition <- matrix(c(0.9, 0.1,
                       0.2, 0.8), nrow = 2, byrow = TRUE)
means <- c(-1, 1.5)
sds <- c(0.7, 0.7)
init <- c(0.5, 0.5)

forward_hmm <- function(y, transition, means, sds, init) {
  T <- length(y)
  K <- length(init)
  alpha <- matrix(0, nrow = T, ncol = K)
  scaling <- numeric(T)

  alpha[1, ] <- init * dnorm(y[1], means, sds)
  scaling[1] <- sum(alpha[1, ])
  alpha[1, ] <- alpha[1, ] / scaling[1]

  for (t in 2:T) {
    for (k in seq_len(K)) {
      alpha[t, k] <- dnorm(y[t], means[k], sds[k]) *
        sum(alpha[t - 1, ] * transition[, k])
    }
    scaling[t] <- sum(alpha[t, ])
    alpha[t, ] <- alpha[t, ] / scaling[t]
  }

  list(alpha = alpha, loglik = sum(log(scaling)))
}

set.seed(2026)
state <- integer(120)
y <- numeric(120)
state[1] <- sample(1:2, 1, prob = init)
y[1] <- rnorm(1, means[state[1]], sds[state[1]])
for (t in 2:120) {
  state[t] <- sample(1:2, 1, prob = transition[state[t - 1], ])
  y[t] <- rnorm(1, means[state[t]], sds[state[t]])
}

fit <- forward_hmm(y, transition, means, sds, init)
print(fit$loglik)
matplot(fit$alpha, type = "l", lwd = 2, lty = 1,
        col = c("firebrick", "steelblue"),
        main = "Filtered HMM state probabilities", ylab = "Probability")
