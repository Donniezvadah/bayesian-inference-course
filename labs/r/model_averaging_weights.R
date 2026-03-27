# Model comparison and averaging for two Gaussian regression models.

set.seed(20260327)

make_design <- function(x, degree) {
  X <- sapply(0:degree, function(p) x^p)
  colnames(X) <- paste0("x", 0:degree)
  X
}

log_marginal_gaussian <- function(y, X, sigma, tau) {
  n <- nrow(X)
  Sigma <- sigma^2 * diag(n) + tau^2 * X %*% t(X)
  chol_Sigma <- chol(Sigma)
  log_det <- 2 * sum(log(diag(chol_Sigma)))
  quad <- drop(t(y) %*% chol2inv(chol_Sigma) %*% y)
  -0.5 * (n * log(2 * pi) + log_det + quad)
}

posterior_gaussian <- function(y, X, sigma, tau) {
  XtX <- crossprod(X)
  V_inv <- XtX / sigma^2 + diag(1 / tau^2, ncol(X))
  V <- solve(V_inv)
  m <- V %*% crossprod(X, y) / sigma^2
  list(mean = drop(m), cov = V)
}

loo_log_predictive <- function(y, X, sigma, tau) {
  n <- nrow(X)
  out <- numeric(n)
  for (i in seq_len(n)) {
    idx <- setdiff(seq_len(n), i)
    post <- posterior_gaussian(y[idx], X[idx, , drop = FALSE], sigma, tau)
    mu <- drop(X[i, , drop = FALSE] %*% post$mean)
    var_pred <- sigma^2 + drop(X[i, , drop = FALSE] %*% post$cov %*% t(X[i, , drop = FALSE]))
    out[i] <- dnorm(y[i], mean = mu, sd = sqrt(var_pred), log = TRUE)
  }
  out
}

stack_objective <- function(eta, log_pred_matrix) {
  w_raw <- c(exp(eta), 1)
  w <- w_raw / sum(w_raw)
  total <- 0
  for (i in seq_len(nrow(log_pred_matrix))) {
    total <- total + log(sum(w * exp(log_pred_matrix[i, ])))
  }
  -total
}

# Simulate mildly nonlinear data so the quadratic model has an advantage.
n <- 80
x <- seq(-1, 1, length.out = n)
y_true <- 1 + 1.5 * x - 2 * x^2
y <- y_true + rnorm(n, sd = 0.35)

sigma <- 0.35
tau <- 2

X1 <- make_design(x, degree = 1)
X2 <- make_design(x, degree = 2)

log_marg_1 <- log_marginal_gaussian(y, X1, sigma, tau)
log_marg_2 <- log_marginal_gaussian(y, X2, sigma, tau)

log_weights <- c(log_marg_1, log_marg_2) - max(log_marg_1, log_marg_2)
bma_weights <- exp(log_weights)
bma_weights <- bma_weights / sum(bma_weights)

loo_matrix <- cbind(
  linear = loo_log_predictive(y, X1, sigma, tau),
  quadratic = loo_log_predictive(y, X2, sigma, tau)
)

opt <- optimize(
  f = stack_objective,
  interval = c(-20, 20),
  log_pred_matrix = loo_matrix
)
stack_weights_raw <- c(exp(opt$minimum), 1)
stack_weights <- stack_weights_raw / sum(stack_weights_raw)
names(stack_weights) <- colnames(loo_matrix)

cat("Exact log marginal likelihoods\n")
print(c(linear = log_marg_1, quadratic = log_marg_2))
cat("\nPosterior model weights (BMA)\n")
print(setNames(round(bma_weights, 4), c("linear", "quadratic")))
cat("\nStacking weights from LOO predictive densities\n")
print(round(stack_weights, 4))
cat("\nMean pointwise LOO log predictive density\n")
print(round(colMeans(loo_matrix), 4))
