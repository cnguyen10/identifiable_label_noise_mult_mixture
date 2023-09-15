import jax
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp

import chex

from functools import partial

from argparse import Namespace


@jax.jit
def batched_logmatmulexp_(logA: chex.Array, logB: chex.Array) -> chex.Array:
    """Implement matrix multiplication in the log scale (see https://stackoverflow.com/a/74409968/5452342)
    Note: this implementation considers a batch of 2-d matrices.

    Args:
        logA: log of the first matrix (N, n, m)
        logB: log of the second matrix  # (N, m, k)

    Returns: log(AB)  # (n, k)
    """
    logA_temp = jnp.tile(A=logA[:, None, :, :], reps=(1, logB.shape[-1], 1, 1))  # (N, k, n, m)
    logA_temp = jnp.transpose(a=logA_temp, axes=(0, 2, 1, 3))  # (N, n, k, m)

    logB_temp = jnp.tile(A=logB[:, None, :, :], reps=(1, logA.shape[-2], 1, 1))  # (N, n, m, k)
    logB_temp = jnp.transpose(a=logB_temp, axes=(0, 1, 3, 2))  # (N, n, k, m)

    logAB = jax.scipy.special.logsumexp(a=logA_temp + logB_temp, axis=-1)  # (N, n, k)

    return logAB


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def EM_for_mm_jit(
    y: chex.Array,
    batch_size_: chex.Numeric,
    n_: chex.Numeric,
    d_: chex.Numeric,
    num_noisy_labels_per_sample: chex.Numeric,
    num_em_iter: chex.Numeric,
    alpha: chex.Numeric,
    beta: chex.Numeric
) -> tuple[chex.Array, chex.Array]:
    """
    """
    # batch_size_, n_, d_ = y.shape

    # region initialize MIXTURE COEFFICIENTS and MULTINOMIAL COMPONENTS
    mixture_coefficients = jnp.array(object=[[1/d_] * d_] * batch_size_)  # (N, C)

    mult_comps_probs = jnp.tile(A=jnp.eye(N=d_), reps=(batch_size_, 1, 1))  # (N, C)

    # add noise to multinomial probabilities

    mult_comps_probs = 10 * mult_comps_probs + jax.random.uniform(key=jax.random.PRNGKey(6870), shape=(batch_size_, d_, d_))  # (N, C, C)
    mult_comps_probs = mult_comps_probs / jnp.sum(a=mult_comps_probs, axis=-1, keepdims=True)  # (N, C, C)
    # endregion

    # initialize a diagonal Dirichlet prior
    log_beta_m1 = jnp.log(beta - 1)

    log_mixture_coefficients_den = jnp.log(n_ + d_ * (alpha - 1))

    log_alpha_m1 = jnp.log(alpha - 1)

    log_mixture_coefficients = jnp.log(mixture_coefficients)
    log_mult_comps_probs = jnp.log(mult_comps_probs)

    for _ in range(num_em_iter):
        multinomial_distributions = tfp.distributions.Multinomial(
            total_count=num_noisy_labels_per_sample,
            logits=log_mult_comps_probs[:, :, None, :]
        )
        log_mult = multinomial_distributions.log_prob(value=y[:, None, :, :])  # (N, C, sample_shape)
        log_mult = jnp.transpose(a=log_mult, axes=(0, 2, 1))  # (N, sample_shape, C)

        # E-step
        log_gamma_num = log_mixture_coefficients[:, None, :] + log_mult  # (N, sample_shape, C)
        log_gamma_den = jax.scipy.special.logsumexp(a=log_gamma_num, axis=-1, keepdims=True)  # (N, sample_shape, 1)
        log_gamma = log_gamma_num - log_gamma_den  # (N, sample_shape, C)

        log_sum_gamma = jax.scipy.special.logsumexp(a=log_gamma, axis=-2, keepdims=False)  # (N, C)

        # M-step
        log_mixture_coefficients_num = jnp.logaddexp(log_sum_gamma, log_alpha_m1)  # (N, C)
        log_mixture_coefficients = log_mixture_coefficients_num - log_mixture_coefficients_den

        log_gamma_y = batched_logmatmulexp_(logA=jnp.transpose(a=log_gamma, axes=(0, 2, 1)), logB=jnp.log(y))  # (N, C, C)
        log_mult_comps_probs_num = jnp.logaddexp(log_gamma_y, log_beta_m1)  # (N, C, C)

        log_mult_comps_probs_den = jnp.log(num_noisy_labels_per_sample) + log_sum_gamma  # (N, C)
        log_mult_comps_probs_den = jnp.logaddexp(log_mult_comps_probs_den, jnp.log(d_) + log_beta_m1)  # (N, C)
        # log_mult_comps_probs_den = log_mult_comps_probs_den + jnp.log1p(d_ * jnp.exp(-log_mult_comps_probs_den))

        log_mult_comps_probs = log_mult_comps_probs_num - log_mult_comps_probs_den[:, :, None]
        # mult_comps_probs = jnp.exp(log_mult_comps_probs)

    return jnp.exp(log_mixture_coefficients), jnp.exp(log_mult_comps_probs)


def EM_for_mm(y: chex.Array, args: Namespace) -> tuple[chex.Array, chex.Array]:
    """Run EM to infer the parameter of a multinomial mixture

    Args:
        y: number of samples  # (N, num_mult_label_sets, C)
        args: an object storing configuration information

    Returns:
        mix_coef:
        mult_comps_probs:
    """
    batch_size_, n_, d_ = y.shape
    return EM_for_mm_jit(
        y=y,
        batch_size_=batch_size_,
        n_=n_,
        d_=d_,
        num_noisy_labels_per_sample=args.num_noisy_labels_per_sample,
        num_em_iter=args.num_em_iter,
        alpha=args.alpha,
        beta=args.beta
    )
