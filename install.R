cran <- "https://cloud.r-project.org"
stan_repo <- c("https://stan-dev.r-universe.dev", cran)

cran_packages <- c(
  "ggplot2",
  "dplyr",
  "tidyr",
  "tibble",
  "posterior",
  "bayesplot",
  "loo",
  "matrixStats"
)

missing_cran <- setdiff(cran_packages, rownames(installed.packages()))
if (length(missing_cran) > 0) {
  install.packages(missing_cran, repos = cran)
}

if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  install.packages("cmdstanr", repos = stan_repo)
}

if (identical(Sys.getenv("INSTALL_CMDSTAN", "false"), "true")) {
  cmdstanr::install_cmdstan()
}
