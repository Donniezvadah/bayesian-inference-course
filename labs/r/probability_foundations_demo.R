# Probability foundations demonstrations:
# LLN, CLT, zero covariance versus dependence, and transformation checks.

set.seed(20260327)

n_grid <- c(5, 20, 100, 500, 2000)
sample_means <- sapply(n_grid, function(n) mean(rexp(n, rate = 1)))
lln_table <- data.frame(n = n_grid, sample_mean = round(sample_means, 4))

# CLT with skewed data: scaled sample means of exponential draws.
S <- 5000
n_clt <- 50
clt_draws <- replicate(S, mean(rexp(n_clt, rate = 1)))
clt_scaled <- sqrt(n_clt) * (clt_draws - 1)

# Dependent but uncorrelated example.
x <- runif(50000, min = -1, max = 1)
y <- x^2
dep_summary <- c(
  mean_x = mean(x),
  mean_y = mean(y),
  covariance = mean((x - mean(x)) * (y - mean(y))),
  correlation = cor(x, y)
)

# Change of variables: Y = exp(X), X ~ N(0,1).
grid_y <- seq(0.05, 4, length.out = 6)
transformation_table <- data.frame(
  y = round(grid_y, 3),
  exact_density = round(dlnorm(grid_y, meanlog = 0, sdlog = 1), 5),
  jacobian_density = round(
    dnorm(log(grid_y), mean = 0, sd = 1) * (1 / grid_y),
    5
  )
)

cat("LLN demonstration with Exponential(1) draws\n")
print(lln_table, row.names = FALSE)

cat("\nCLT demonstration: scaled sample means of Exponential(1) draws\n")
print(round(c(
  mean = mean(clt_scaled),
  sd = sd(clt_scaled),
  q025 = unname(quantile(clt_scaled, 0.025)),
  q975 = unname(quantile(clt_scaled, 0.975))
), 4))

cat("\nDependent but nearly uncorrelated example Y = X^2 with X ~ Unif(-1,1)\n")
print(round(dep_summary, 5))

cat("\nTransformation check for Y = exp(X), X ~ N(0,1)\n")
print(transformation_table, row.names = FALSE)
