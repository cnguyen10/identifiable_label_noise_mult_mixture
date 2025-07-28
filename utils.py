import jax
from jax import numpy as jnp
from jax.experimental import sparse

from jaxopt import OSQP
import optax

from tensorflow_probability.substrates import jax as tfp

import faiss  # facebook similarity search

import numpy as np

from functools import partial

import grain.python as grain

from transformations import (
    RandomCrop,
    Resize,
    CropAndPad,
    RandomHorizontalFlip,
    ToRGB,
    Normalize,
    ToFloat
)


@partial(jax.jit, static_argnums=(2, 3))
def batched_logmatmulexp_(logA: jax.Array, logB: jax.Array, num_rows_A: int, num_cols_B: int) -> jax.Array:
    """Implement matrix multiplication in the log scale (see https://stackoverflow.com/a/74409968/5452342)
    Note: this implementation considers a batch of 2-d matrices.

    Args:
        logA: log of the first matrix (N, n, m)
        logB: log of the second matrix  # (N, m, k)
        num_rows_A: the number of rows in matrix A (or n in this case)
        num_cols_B: the number of columns in matrix B (or k in this case)

    Returns: log(AB)  # (n, k)
    """
    logA_temp = jnp.tile(A=jnp.expand_dims(a=logA, axis=1), reps=(1, num_cols_B, 1, 1))  # (N, k, n, m)
    logA_temp = jnp.transpose(a=logA_temp, axes=(0, 2, 1, 3))  # (N, n, k, m)

    logB_temp = jnp.tile(A=jnp.expand_dims(a=logB, axis=1), reps=(1, num_rows_A, 1, 1))  # (N, n, m, k)
    logB_temp = jnp.transpose(a=logB_temp, axes=(0, 1, 3, 2))  # (N, n, k, m)

    logAB = jax.scipy.special.logsumexp(a=logA_temp + logB_temp, axis=-1)  # (N, n, k)

    return logAB


@partial(
    jax.jit,
    static_argnames=(
        'batch_size_',
        'n_',
        'd_',
        'num_noisy_labels_per_sample',
        'num_em_iter',
        'alpha',
        'beta'
    )
)
def EM_for_mm(
    y: jax.Array,
    batch_size_: int,
    n_: int,
    d_: int,
    num_noisy_labels_per_sample: int,
    num_em_iter: int,
    alpha: float,
    beta: float,
    key: jax._src.prng.PRNGKeyArray
) -> tuple[jax.Array, jax.Array]:
    """
    Args:
        y: an array of data (N, num_multinomial_samples, C)
        n_: the number of multinomial samples
        d_: the number of classes/categories in the multinomial distributions
        num_noisy_labels_per_sample: the total number of trials in each multinomial sample
        num_em_iter:
        alpha: the Dirichlet prior parameter for mixture coefficients
        beta: the Dirichlet prior parameter for multinomial components

    Returns:
        log_mixture_coefficients:
        log_mult_comps_probs:
    """
    # region initialize MIXTURE COEFFICIENTS and MULTINOMIAL COMPONENTS
    mixture_coefficients = jnp.array(object=[[1/d_] * d_] * batch_size_)  # (N, C)

    mult_comps_probs = jnp.tile(A=jnp.eye(N=d_), reps=(batch_size_, 1, 1))  # (N, C)

    # add noise to multinomial probabilities

    mult_comps_probs = 10 * mult_comps_probs + jax.random.uniform(key=key, shape=(batch_size_, d_, d_))  # (N, C, C)
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
            logits=jnp.expand_dims(a=log_mult_comps_probs, axis=-2)
        )
        log_mult = multinomial_distributions.log_prob(
            value=jnp.expand_dims(a=y, axis=1)
        )  # (N, C, sample_shape)
        log_mult = jnp.transpose(a=log_mult, axes=(0, 2, 1))  # (N, sample_shape, C)

        # E-step
        log_gamma_num = jnp.expand_dims(a=log_mixture_coefficients, axis=1) + log_mult  # (N, sample_shape, C)
        log_gamma_den = jax.scipy.special.logsumexp(a=log_gamma_num, axis=-1, keepdims=True)  # (N, sample_shape, 1)
        log_gamma = log_gamma_num - log_gamma_den  # (N, sample_shape, C)

        log_sum_gamma = jax.scipy.special.logsumexp(a=log_gamma, axis=-2, keepdims=False)  # (N, C)

        # M-step
        log_mixture_coefficients_num = jnp.logaddexp(log_sum_gamma, log_alpha_m1)  # (N, C)
        log_mixture_coefficients = log_mixture_coefficients_num - log_mixture_coefficients_den

        log_gamma_y = batched_logmatmulexp_(
            logA=jnp.transpose(a=log_gamma, axes=(0, 2, 1)),
            logB=jnp.log(y),
            num_rows_A=d_,
            num_cols_B=d_
        )  # (N, C, C)
        log_mult_comps_probs_num = jnp.logaddexp(log_gamma_y, log_beta_m1)  # (N, C, C)

        log_mult_comps_probs_den = jnp.log(num_noisy_labels_per_sample) + log_sum_gamma  # (N, C)
        log_mult_comps_probs_den = jnp.logaddexp(log_mult_comps_probs_den, jnp.log(d_) + log_beta_m1)  # (N, C)

        log_mult_comps_probs = log_mult_comps_probs_num - jnp.expand_dims(a=log_mult_comps_probs_den, axis=-1)

    return log_mixture_coefficients, log_mult_comps_probs


def sub_sparse_matrix_from_row_indices(mat: sparse.BCOO, indices: jax.Array, n_batch: int = 1) -> sparse.BCOO:
    """
    """
    m = mat[indices]
    m = sparse.bcoo_update_layout(mat=m, n_batch=n_batch)
    m = sparse.bcoo_sum_duplicates(mat=m)

    return m


def get_knn_indices(xb: np.ndarray, num_nn: int) -> np.ndarray:
    """find the indices of k-nearest neighbours

    Args:
        xb: the matrix where each row contains feature vector of one datum
        args:

    Returns:
        index_matrix: matrix containing indices of nearest neighbours
    """
    res = faiss.StandardGpuResources()  # use a single GPU

    index_flat = faiss.IndexFlatL2(xb.shape[-1])  # build a flat (CPU) index
    index_id_map = faiss.IndexIDMap(index_flat)  # translates ids when adding and searching
    gpu_index_flat = faiss.index_cpu_to_gpu(provider=res, device=0, index=index_id_map)  # make it into a gpu index

    ids = np.arange(stop=xb.shape[0])

    gpu_index_flat.add_with_ids(xb, ids)  # add vectors to the index

    # start the nearest-neighbour search
    _, index_matrix = gpu_index_flat.search(x=xb, k=num_nn + 1)  # adding 1 is because the return includes the sample itself

    return index_matrix[:, 1:]  # exclude the first element which is the sample itself


def solve_local_affine_coding(datum: jax.Array, knn: jax.Array) -> jax.Array:
    """A JAX-jittable method to calculate the local affine coding of a single sample
    from its nearest neighbours
    The optimization is an operator-splitted quadratic program (OSQP):
    min 0.5 * x^T Q x + c^T x subject to Gx <= h, Ax = b.

    Args:
        datum: the data vector of the sample  # (d,)
        knn: the matrix where each row consists of K nearest-neighbour indices  # (K, d)

    Returns:
        x: the coding vector
    """
    matrix_Q: jax.Array
    vector_c: jax.Array
    matrix_G: jax.Array
    vector_h: jax.Array
    matrix_A: jax.Array
    vector_b: jax.Array

    # parameters of objective functions
    matrix_Q = jnp.matmul(a=knn, b=jnp.transpose(a=knn))  # (K, K)
    vector_c = - jnp.matmul(a=knn, b=datum)  # (K,)

    # inequality
    matrix_G = -jnp.identity(n=knn.shape[0])  # (K, K)
    vector_h = jnp.zeros_like(a=vector_c)  # (K,)

    # equality
    matrix_A = jnp.ones_like(a=vector_c)[None, :]  # (K,)
    vector_b = jnp.array(object=[1.])  # scalar

    # declare the OSQP object
    qp = OSQP(maxiter=1000, tol=1e-6)
    sol = qp.run(params_obj=(matrix_Q, vector_c), params_ineq=(matrix_G, vector_h), params_eq=(matrix_A, vector_b))

    x = sol.params.primal

    # the OSQP stops at a certain point and the solution might be closed,
    # but not exact. Thus, we need to enforce the constrains:
    # x = jnp.abs(x)  # non-negative number
    x = jax.nn.relu(x)
    x = x / jnp.sum(a=x, axis=-1)  # sum to 1

    return x


def get_local_affine_coding(datum: jax.Array, knn_index: jax.Array, data: jax.Array) -> jax.Array:
    """A JAX-based method to calculate the local affine coding of a single sample
    from its nearest neighbours
    The optimization is an operator-splitted quadratic program (OSQP):
    min 0.5 * x^T Q x + c^T x subject to Gx <= h, Ax = b.

    Args:
        datum: the data vector of the sample  # (d,)
        knn_index: vector of K nearest-neighbour indices  # (K,)
        data: all the samples considered  (N, d)

    Returns:
        x: the coding vector
    """
    # get K nearest-neighbours
    knn = data[knn_index]  # (K, d)

    return solve_local_affine_coding(datum=datum, knn=knn)


def get_batch_local_affine_coding(samples: np.ndarray, knn_indices: np.ndarray) -> jax.Array:
    """This method calculates the local affine coding of a batch of samples
    The optimization is an operator-splitted quadratic program (OSQP):
    min 0.5 * x^T Q x + c^T x subject to Gx <= h, Ax = b.

    Args:
        samples: a batch of data vectors  # (N, d)
        knn: the tensors of K nearest-neighbour of the batch of samples  # (N, K, d)

    Returns:
        x: the batch of coding vectors  # (N, K)
    """
    # convert the matrix of nearest-neighbours to jax array
    samples_jax = jnp.asarray(a=samples)  # (N, d)
    knn_indices_jax = jnp.asarray(a=knn_indices)  # (N, K, d)

    get_LAC_partial = partial(get_local_affine_coding, data=samples_jax)
    auto_batch_LAC = jax.vmap(fun=get_LAC_partial, in_axes=0, out_axes=0)

    x = auto_batch_LAC(samples_jax, knn_indices_jax)  # (N, K)

    return x


def init_tx(
    dataset_length: int,
    lr: float,
    batch_size: int,
    num_epochs: int,
    weight_decay: float,
    momentum: float,
    clipped_norm: float,
    key: jax._src.prng.PRNGKeyArray
) -> optax.GradientTransformationExtraArgs:
    """initialize parameters of an optimizer
    """
    # add L2 regularization(a.k.a. weight decay)
    l2_regularization = optax.masked(
        inner=optax.add_decayed_weights(
            weight_decay=weight_decay,
            mask=None
        ),
        mask=lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
    )

    num_iters_per_epoch = dataset_length // batch_size
    lr_schedule_fn = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=(num_epochs + 10) * num_iters_per_epoch
    )

    # define an optimizer
    tx = optax.chain(
        l2_regularization,
        optax.clip_by_global_norm(max_norm=clipped_norm) \
            if clipped_norm is not None else optax.identity(),
        optax.add_noise(eta=0.01, gamma=0.55, key=key),
        optax.sgd(learning_rate=lr_schedule_fn, momentum=momentum)
    )

    return tx


def initialize_dataloader(
    data_source: grain.RandomAccessDataSource,
    num_epochs: int,
    shuffle: bool,
    seed: int,
    batch_size: int,
    drop_remainder: bool = False,
    crop_size: tuple[int, int] | None = None,
    padding_px: int | list[int] | None = None,
    resize: tuple[int, int] | None = None,
    mean: float | None = None,
    std: float | None = None,
    p_flip: float | None = None,
    is_color_img: bool = True,
    num_workers: int = 0,
    num_threads: int = 1,
    prefetch_size: int = 1
) -> grain.DataLoader:
    """
    """
    index_sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=num_epochs,
        shuffle=shuffle,
        shard_options=grain.NoSharding(),
        seed=seed  # set the random seed
    )

    transformations = []

    if resize is not None:
        transformations.append(Resize(resize_shape=resize))
    
    if padding_px is not None:
        transformations.append(CropAndPad(px=padding_px))

    if crop_size is not None:
        transformations.append(RandomCrop(crop_size=crop_size))

    if p_flip is not None:
        transformations.append(RandomHorizontalFlip(p=p_flip))
        # transformations.append(RandomVerticalFlip(p=p_flip))

    if not is_color_img:
        transformations.append(ToRGB())

    transformations.append(ToFloat())

    if mean is not None and std is not None:
        transformations.append(Normalize(mean=mean, std=std))

    transformations.append(
        grain.Batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder
        )
    )

    data_loader = grain.DataLoader(
        data_source=data_source,
        sampler=index_sampler,
        operations=transformations,
        worker_count=num_workers,
        shard_options=grain.NoSharding(),
        read_options=grain.ReadOptions(
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_size
        )
    )

    return data_loader