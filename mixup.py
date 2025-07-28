import jax
import jax.numpy as jnp


@jax.jit
def mixup_data(
    x: jax.Array,
    y: jax.Array,
    key: jax._src.prng.PRNGKeyArray,
    beta_a: jax.typing.ArrayLike,
    beta_b: jax.typing.ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """apply mixup in training

    Args:
        x: input samples
        y: one-hot labels
        key: random key
        beta_a, beta_b: parameters of the Beta distribution

    Returns:
        x_mixed: the mixup input samples
        y_mixed: the mixup label vectors
    """
    shuffled_ids = jax.random.permutation(key=key, x=jnp.arange(len(y)))
    x2 = x[shuffled_ids]
    y2 = y[shuffled_ids]

    mixup_ratio = jax.random.beta(
        key=key,
        a=beta_a,
        b=beta_b,
        shape=(1,)
    )

    x_mixed = mixup_ratio * x + (1 - mixup_ratio) * x2
    y_mixed = mixup_ratio * y + (1 - mixup_ratio) * y2

    return (x_mixed, y_mixed)