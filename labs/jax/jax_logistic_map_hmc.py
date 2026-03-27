"""Bayesian logistic regression in pure JAX.

This script demonstrates:
1. Log posterior evaluation
2. Automatic differentiation
3. Newton MAP optimization
4. Laplace approximation
5. A minimal HMC-style sampler
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def simulate_data(key, n=120):
    key_x, key_noise = jax.random.split(key)
    x = jax.random.normal(key_x, shape=(n,))
    X = jnp.column_stack((jnp.ones(n), x))
    beta_true = jnp.array([-0.4, 1.2])
    eta = X @ beta_true
    probs = jax.nn.sigmoid(eta)
    y = jax.random.bernoulli(key_noise, probs).astype(jnp.float64)
    return X, y, beta_true


def log_posterior(beta, X, y, prior_scale=2.5):
    eta = X @ beta
    loglik = jnp.sum(y * eta - jax.nn.softplus(eta))
    logprior = -0.5 * jnp.sum((beta / prior_scale) ** 2)
    return loglik + logprior


def newton_map(beta_init, X, y, prior_scale=2.5, n_steps=8):
    def neg_log_post(beta):
        return -log_posterior(beta, X, y, prior_scale)

    grad_u = jax.grad(neg_log_post)
    hess_u = jax.hessian(neg_log_post)
    beta = beta_init

    for _ in range(n_steps):
        g = grad_u(beta)
        H = hess_u(beta)
        step = jnp.linalg.solve(H, g)
        beta = beta - step

    return beta


def leapfrog(position, momentum, step_size, n_steps, X, y, prior_scale=2.5):
    def potential(q):
        return -log_posterior(q, X, y, prior_scale)

    grad_u = jax.grad(potential)
    q = position
    p = momentum - 0.5 * step_size * grad_u(q)

    for step in range(n_steps):
        q = q + step_size * p
        if step != n_steps - 1:
            p = p - step_size * grad_u(q)

    p = p - 0.5 * step_size * grad_u(q)
    return q, -p


def hmc_step(key, position, step_size, n_steps, X, y, prior_scale=2.5):
    momentum_key, accept_key = jax.random.split(key)
    momentum = jax.random.normal(momentum_key, shape=position.shape)

    def potential(q):
        return -log_posterior(q, X, y, prior_scale)

    def kinetic(p):
        return 0.5 * jnp.dot(p, p)

    proposed_q, proposed_p = leapfrog(
        position, momentum, step_size, n_steps, X, y, prior_scale
    )

    current_h = potential(position) + kinetic(momentum)
    proposed_h = potential(proposed_q) + kinetic(proposed_p)
    log_alpha = current_h - proposed_h

    accept = jnp.log(jax.random.uniform(accept_key)) < log_alpha
    new_position = jnp.where(accept, proposed_q, position)
    return new_position, bool(accept)


def run_hmc(key, init, n_samples, step_size, n_steps, X, y, prior_scale=2.5):
    draws = []
    accepts = 0
    position = init

    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        position, accepted = hmc_step(
            subkey, position, step_size, n_steps, X, y, prior_scale
        )
        draws.append(position)
        accepts += int(accepted)

    return jnp.stack(draws), accepts / n_samples


def main():
    key = jax.random.PRNGKey(7)
    X, y, beta_true = simulate_data(key)
    beta_init = jnp.zeros(X.shape[1])

    beta_map = newton_map(beta_init, X, y)
    hessian = jax.hessian(lambda b: -log_posterior(b, X, y))(beta_map)
    cov_laplace = jnp.linalg.inv(hessian)

    key, hmc_key = jax.random.split(key)
    draws, accept_rate = run_hmc(
        hmc_key,
        init=beta_map,
        n_samples=2000,
        step_size=0.03,
        n_steps=25,
        X=X,
        y=y
    )

    posterior_mean = draws.mean(axis=0)
    posterior_sd = draws.std(axis=0)

    print("True beta:", beta_true)
    print("MAP estimate:", beta_map)
    print("Laplace covariance:\n", cov_laplace)
    print("HMC acceptance rate:", accept_rate)
    print("Posterior mean:", posterior_mean)
    print("Posterior SD:", posterior_sd)


if __name__ == "__main__":
    main()
