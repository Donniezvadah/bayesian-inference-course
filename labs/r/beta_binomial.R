# Beta-Binomial updating and posterior prediction

beta_binomial_update <- function(y, n, alpha, beta) {
  alpha_post <- alpha + y
  beta_post <- beta + n - y

  list(
    alpha_post = alpha_post,
    beta_post = beta_post,
    post_mean = alpha_post / (alpha_post + beta_post),
    post_var = (alpha_post * beta_post) /
      (((alpha_post + beta_post)^2) * (alpha_post + beta_post + 1))
  )
}

beta_binomial_predictive <- function(k, m, y, n, alpha, beta) {
  choose(m, k) *
    beta(alpha + y + k, beta + n - y + m - k) /
    beta(alpha + y, beta + n - y)
}

beta_binomial_marginal_likelihood <- function(y, n, alpha, beta) {
  choose(n, y) * beta(alpha + y, beta + n - y) / beta(alpha, beta)
}

set.seed(123)
y <- 14
n <- 20
alpha <- 2
beta <- 2

posterior <- beta_binomial_update(y, n, alpha, beta)
print(posterior)

pred_counts <- 0:10
pred_probs <- sapply(
  pred_counts,
  beta_binomial_predictive,
  m = 10,
  y = y,
  n = n,
  alpha = alpha,
  beta = beta
)

print(data.frame(k = pred_counts, predictive_prob = pred_probs))

grid <- seq(0.001, 0.999, length.out = 400)
prior_density <- dbeta(grid, alpha, beta)
post_density <- dbeta(grid, posterior$alpha_post, posterior$beta_post)

plot(
  grid, prior_density,
  type = "l",
  lwd = 2,
  col = "steelblue",
  ylab = "Density",
  xlab = expression(theta),
  main = "Beta-Binomial updating"
)
lines(grid, post_density, lwd = 2, col = "firebrick")
legend(
  "topright",
  legend = c("Prior", "Posterior"),
  col = c("steelblue", "firebrick"),
  lwd = 2,
  bty = "n"
)
