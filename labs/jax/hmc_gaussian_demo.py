"""Minimal HMC demonstration on a correlated Gaussian target in JAX."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def potential(q, sigma_inv):
    return 0.5 * q.T @ sigma_inv @ q


def kinetic(p):
    return 0.5 * jnp.dot(p, p)


def grad_potential(q, sigma_inv):
    return sigma_inv @ q


def leapfrog(q, p, epsilon, n_steps, sigma_inv):
    p = p - 0.5 * epsilon * grad_potential(q, sigma_inv)

    for step in range(n_steps):
        q = q + epsilon * p
        if step != n_steps - 1:
            p = p - epsilon * grad_potential(q, sigma_inv)

    p = p - 0.5 * epsilon * grad_potential(q, sigma_inv)
    return q, -p


def hmc_step(key, q, epsilon, n_steps, sigma_inv):
    key_mom, key_acc = jax.random.split(key)
    p = jax.random.normal(key_mom, shape=q.shape)
    q_prop, p_prop = leapfrog(q, p, epsilon, n_steps, sigma_inv)

    current_h = potential(q, sigma_inv) + kinetic(p)
    proposed_h = potential(q_prop, sigma_inv) + kinetic(p_prop)
    log_alpha = current_h - proposed_h

    accept = jnp.log(jax.random.uniform(key_acc)) < log_alpha
    q_new = jnp.where(accept, q_prop, q)
    return q_new, bool(accept)


def run_hmc(key, n_samples=2000, epsilon=0.12, n_steps=15, rho=0.95):
    sigma = jnp.array([[1.0, rho], [rho, 1.0]])
    sigma_inv = jnp.linalg.inv(sigma)

    q = jnp.array([0.0, 0.0])
    draws = []
    accepts = 0

    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        q, accepted = hmc_step(subkey, q, epsilon, n_steps, sigma_inv)
        draws.append(q)
        accepts += int(accepted)

    draws = jnp.stack(draws)
    return draws, accepts / n_samples


def main():
    key = jax.random.PRNGKey(123)
    draws, accept_rate = run_hmc(key)

    print("Acceptance rate:", accept_rate)
    print("Posterior mean estimate:", draws.mean(axis=0))
    print("Empirical covariance:\n", jnp.cov(draws.T))


if __name__ == "__main__":
    main()
