# Telescoping sampler for a univariate normal mixture with unknown K.
#
# Implements the generalised mixture of finite mixtures of
# Fruhwirth-Schnatter, Malsiner-Walli and Grun (2021), Bayesian Analysis 16(4).
#
# The number of components K is drawn from its conditional posterior given the
# partition; components are then added by drawing from the prior and removed by
# discarding empty ones. No reversible-jump machinery is required.

# ---------------------------------------------------------------------------
# Prior on the number of components
# ---------------------------------------------------------------------------

# Beta-negative-binomial prior on K - 1. BNB(1, 4, 3) is the recommended
# default: heavy enough in the tail to let the data raise K, with mass
# concentrated on small K.
log_prior_K <- function(K, a_lambda = 1, a_pi = 4, b_pi = 3) {
  x <- K - 1
  lbeta(a_lambda + x, a_pi + b_pi) - lbeta(a_lambda, a_pi) +
    lgamma(x + b_pi) - lgamma(x + 1) - lgamma(b_pi)
}

# Poisson alternative, for prior sensitivity checks.
log_prior_K_pois <- function(K, lambda = 1) {
  dpois(K - 1, lambda, log = TRUE)
}

# ---------------------------------------------------------------------------
# The telescoping step
# ---------------------------------------------------------------------------

# Log conditional posterior of K given the partition, up to an additive
# constant. N_k holds the sizes of the K_plus filled blocks; the component
# parameters and the observations do not enter.
log_post_K <- function(K, N_k, N, alpha, dynamic = TRUE,
                       log_prior = log_prior_K) {
  K_plus <- length(N_k)
  if (K < K_plus) return(-Inf)

  gamma_K <- if (dynamic) alpha / K else alpha

  # log of the falling factorial K! / (K - K_plus)!, computed in logs so that
  # large K does not overflow.
  lfalling <- sum(log(seq.int(K - K_plus + 1, K)))

  lfalling +
    lgamma(K * gamma_K) - lgamma(N + K * gamma_K) +
    sum(lgamma(N_k + gamma_K)) - K_plus * lgamma(gamma_K) +
    log_prior(K)
}

sample_K <- function(N_k, N, alpha, K_max = 100, dynamic = TRUE,
                     log_prior = log_prior_K) {
  K_plus <- length(N_k)
  grid <- K_plus:K_max
  lp <- vapply(grid, log_post_K, numeric(1),
               N_k = N_k, N = N, alpha = alpha, dynamic = dynamic,
               log_prior = log_prior)
  p <- exp(lp - max(lp))
  if (grid[which.max(p)] == K_max) {
    warning("posterior mass at K_max; increase K_max")
  }
  sample(grid, size = 1, prob = p)
}

# ---------------------------------------------------------------------------
# Full sampler
# ---------------------------------------------------------------------------

telescoping_mixture <- function(y, n_iter = 10000, alpha = 1,
                                K_init = 5, K_max = 100, dynamic = TRUE,
                                m0 = mean(y), kappa0 = 0.01,
                                a0 = 2, b0 = var(y) / 2,
                                log_prior = log_prior_K) {
  N <- length(y)
  K <- K_init
  mu <- rnorm(K, m0, sd(y))
  sigma2 <- rep(var(y), K)
  eta <- rep(1 / K, K)

  keep_K <- integer(n_iter)
  keep_K_plus <- integer(n_iter)
  keep_S <- matrix(0L, nrow = n_iter, ncol = N)

  for (it in seq_len(n_iter)) {

    ## Step 1: allocations
    logdens <- vapply(seq_len(K), function(k) {
      log(eta[k]) + dnorm(y, mu[k], sqrt(sigma2[k]), log = TRUE)
    }, numeric(N))
    logdens <- logdens - apply(logdens, 1, max)
    prob <- exp(logdens)
    S <- apply(prob, 1, function(p) sample.int(K, 1, prob = p))

    ## Relabel so that filled components come first, then drop empties
    filled <- sort(unique(S))
    K_plus <- length(filled)
    S <- match(S, filled)
    mu <- mu[filled]
    sigma2 <- sigma2[filled]
    N_k <- tabulate(S, nbins = K_plus)

    ## Step 2: filled component parameters, conjugate normal-inverse-gamma
    for (k in seq_len(K_plus)) {
      yk <- y[S == k]
      nk <- length(yk)
      ybar <- mean(yk)
      kappa_n <- kappa0 + nk
      m_n <- (kappa0 * m0 + sum(yk)) / kappa_n
      a_n <- a0 + nk / 2
      b_n <- b0 + 0.5 * sum((yk - ybar)^2) +
        0.5 * kappa0 * nk * (ybar - m0)^2 / kappa_n
      sigma2[k] <- 1 / rgamma(1, a_n, b_n)
      mu[k] <- rnorm(1, m_n, sqrt(sigma2[k] / kappa_n))
    }

    ## Step 5: telescope
    K <- sample_K(N_k, N, alpha, K_max = K_max, dynamic = dynamic,
                  log_prior = log_prior)
    gamma_K <- if (dynamic) alpha / K else alpha

    ## Step 6: refill from the prior, redraw weights
    n_new <- K - K_plus
    if (n_new > 0) {
      sigma2_new <- 1 / rgamma(n_new, a0, b0)
      mu_new <- rnorm(n_new, m0, sqrt(sigma2_new / kappa0))
      mu <- c(mu, mu_new)
      sigma2 <- c(sigma2, sigma2_new)
    }
    conc <- c(N_k + gamma_K, rep(gamma_K, n_new))
    g <- rgamma(K, shape = conc, rate = 1)
    eta <- g / sum(g)

    keep_K[it] <- K
    keep_K_plus[it] <- K_plus
    keep_S[it, ] <- S
  }

  list(K = keep_K, K_plus = keep_K_plus, S = keep_S)
}

# ---------------------------------------------------------------------------
# Posterior summaries over partitions
# ---------------------------------------------------------------------------

# Proportion of draws in which observations i and j share a component.
# Invariant to relabelling, so it sidesteps label switching entirely.
similarity_matrix <- function(S_draws) {
  N <- ncol(S_draws)
  M <- nrow(S_draws)
  Pi <- matrix(0, N, N)
  for (m in seq_len(M)) {
    Pi <- Pi + outer(S_draws[m, ], S_draws[m, ], "==")
  }
  Pi / M
}

# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

if (sys.nframe() == 0L) {
  set.seed(2024)
  y <- c(rnorm(120, -3, 1), rnorm(80, 0, 0.7), rnorm(100, 4, 1.2))

  fit <- telescoping_mixture(y, n_iter = 20000, alpha = 1)
  draws <- 5001:20000

  cat("Posterior of K (number of components):\n")
  print(round(prop.table(table(fit$K[draws])), 3))

  cat("\nPosterior of K_plus (number of occupied components):\n")
  print(round(prop.table(table(fit$K_plus[draws])), 3))

  # Prior sensitivity: the same data under a Poisson(1) prior on K - 1.
  fit_pois <- telescoping_mixture(y, n_iter = 20000, alpha = 1,
                                  log_prior = log_prior_K_pois)
  cat("\nPosterior of K_plus under a Poisson prior:\n")
  print(round(prop.table(table(fit_pois$K_plus[draws])), 3))

  Pi <- similarity_matrix(fit$S[draws, ])
  cat("\nMean within-group co-clustering probability:",
      round(mean(Pi[1:120, 1:120]), 3), "\n")
}
