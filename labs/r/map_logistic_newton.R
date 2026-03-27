# MAP estimation for Bayesian logistic regression using Newton updates.

set.seed(2026)
n <- 250
x <- rnorm(n)
X <- cbind(1, x)
beta_true <- c(-0.4, 1.0)
p <- plogis(drop(X %*% beta_true))
y <- rbinom(n, 1, p)

neg_log_post <- function(beta, X, y, prior_sd = 2.5) {
  eta <- drop(X %*% beta)
  -sum(y * eta - log1p(exp(eta))) + sum(beta^2) / (2 * prior_sd^2)
}

grad_neg_log_post <- function(beta, X, y, prior_sd = 2.5) {
  eta <- drop(X %*% beta)
  p <- plogis(eta)
  drop(crossprod(X, p - y) + beta / prior_sd^2)
}

hess_neg_log_post <- function(beta, X, y, prior_sd = 2.5) {
  eta <- drop(X %*% beta)
  p <- plogis(eta)
  W <- diag(p * (1 - p))
  crossprod(X, W %*% X) + diag(1 / prior_sd^2, ncol(X))
}

beta <- c(0, 0)
for (iter in 1:10) {
  g <- grad_neg_log_post(beta, X, y)
  H <- hess_neg_log_post(beta, X, y)
  beta <- beta - solve(H, g)
}

cat("MAP estimate:\n")
print(beta)
