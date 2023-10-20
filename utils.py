import jax
from jax import numpy as jnp
import jax.dlpack
from jax.experimental import sparse

import flax
from flax import linen as nn
from flax.training import train_state

from jaxopt import OSQP
import optax

from clu import metrics

import dm_pix

from tensorflow_probability.substrates import jax as tfp

import chex

import tensorflow_datasets as tfds
import tensorflow as tf

import faiss  # facebook similarity search

import numpy as np
import os

from functools import partial

import aim

import argparse

from typing import Any, Callable, NamedTuple
from tqdm import tqdm

from data_utils import image_folder_label_csv


@partial(jax.jit, static_argnums=(2, 3))
def batched_logmatmulexp_(logA: chex.Array, logB: chex.Array, num_rows_A: int, num_cols_B: int) -> chex.Array:
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


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def EM_for_mm(
    y: chex.Array,
    batch_size_: chex.Numeric,
    n_: chex.Numeric,
    d_: chex.Numeric,
    num_noisy_labels_per_sample: chex.Numeric,
    num_em_iter: chex.Numeric,
    alpha: chex.Numeric,
    beta: chex.Numeric,
    key: jax.random.KeyArray
) -> tuple[chex.Array, chex.Array]:
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


@partial(jax.jit, static_argnums=(2,))
def augment_an_image(key: jax.random.PRNGKey, image: chex.Array, image_shape: tuple[int, int, int]) -> chex.Array:
    """perform data augmentation on a single image. For a batch of images,
    please apply jax.vmap

    Args:
        key: the random key for PRNG
        image: the image of interest
        image_shape: a tuple of image shape (must be tuple, not list)

    Returns:
        image: the augmented image
    """
    image = dm_pix.pad_to_size(
        image=image,
        target_height=image.shape[0] + 4,
        target_width=image.shape[1] + 4
    )
    image = dm_pix.random_crop(key=key, image=image, crop_sizes=image_shape)
    image = dm_pix.random_flip_left_right(key=key, image=image)

    return image


def sub_sparse_matrix_from_row_indices(mat: sparse.BCOO, indices: chex.Array, n_batch: int = 1) -> sparse.BCOO:
    """
    """
    m = mat[indices]
    m = sparse.bcoo_update_layout(mat=m, n_batch=n_batch)
    m = sparse.bcoo_sum_duplicates(mat=m)

    return m


def parse_arguments() -> argparse.Namespace:
    """
    """
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--experiment-name', type=str, help='')

    parser.add_argument('--dataset-root', type=str, default=None, help='Path to the folder containing train and test sets')
    parser.add_argument('--logdir', type=str, default='logs', help='Folder to store logs')

    parser.add_argument('--img-shape', action='append', help='e.g., 32 32 3 or 224 224 3')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples in each iteration. None is whole dataset')
    parser.add_argument('--C0', type=int, help='Number of sparse classes')

    parser.add_argument('--label-filenames', action='append', help='Filenames of images and labels in the split folder')

    parser.add_argument('--k', type=int, help='Number of nearest neighbours')
    parser.add_argument('--num-noisy-labels-per-sample', type=int, help='Number of noisy labels per training sample')
    parser.add_argument('--num-multinomial-samples', type=int, default=5_000, help='Number of multinomial samples used in EM')

    # region EM-related
    parser.add_argument('--mu', type=float, help='Percentage between nearest neighbors and itself')

    parser.add_argument('--batch-size-em', type=int, help='Batch size of samples are EM-ed simultaneously')

    parser.add_argument('--alpha', type=float, default=1., help='Dirichlet prior parameter for p_y')
    parser.add_argument('--beta', type=float, default=1., help='Dirichlet prior parameter for p(y_hat | x, y)')
    parser.add_argument('--num-em-iter', type=int, default=5, help='Number of EM iterations')
    # endregion

    # region PYTORCH
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num-warmup', type=int, help='Number of epochs to warm up')
    parser.add_argument('--num-epochs', type=int, help='Number of epochs')
    # endregion

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument('--tqdm', dest='tqdm_flag', action='store_true')
    parser.add_argument('--no-tqdm', dest='tqdm_flag', action='store_false')
    parser.set_defaults(tqdm_flag=True)

    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.set_defaults(resume=False)
    parser.add_argument('--run-hash-id', type=str, default=None, help='Hash id of the run to resume')

    parser.add_argument('--jax-mem-fraction', type=float, default=0.1, help='Percentage of GPU memory allocated for Jax')

    args = parser.parse_args()

    return args


@jax.jit
def feature_step(state: train_state.TrainState, x: chex.Array) -> chex.Array:
    """
    """
    features, _ = state.apply_fn(
        variables={'params': state.params, 'batch_stats': state.batch_stats},
        x=x,
        mutable=['batch_stats'],
        method=lambda module, x: module.features(x=x, train=False)
    )

    return features


def get_features(state: train_state.TrainState, ds: tf.data.Dataset, batch_size: chex.Numeric) -> chex.Array:
    """Extract features of data

    Args:
        state: a data-class storing model-related parameters
        ds: the dataset of interest
        batch_size:

    Returns:
        xb: the extracted features
    """
    # get feature dimension
    for x, _ in tfds.as_numpy(dataset=ds):
        x = jax.device_put(x=x) / 255.
        features = feature_step(state, x)
        break

    xb = jnp.empty(shape=(len(ds), features.shape[-1]))  # (N, D)

    ds = ds.batch(batch_size=batch_size)

    # region EXTRACT FEATURES
    start_idx = 0
    for x, _ in tqdm(
        iterable=tfds.as_numpy(dataset=ds),
        desc=' features',
        leave=False,
        position=1
    ):
        x = jax.device_put(x=x) / 255.
        features = feature_step(state, x)

        end_idx = start_idx + x.shape[0]
        xb = xb.at[start_idx:end_idx].set(features)
        start_idx = end_idx
    # endregion

    return xb


def get_knn_indices(xb: np.ndarray, num_nn: chex.Numeric) -> np.ndarray:
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


def solve_local_affine_coding(datum: chex.Array, knn: chex.Array) -> jax.Array:
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


def get_local_affine_coding(datum: chex.Array, knn_index: chex.Array, data: chex.Array) -> chex.Array:
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


def get_batch_local_affine_coding(samples: np.ndarray, knn_indices: np.ndarray) -> chex.Array:
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


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def _get_p_y(
    log_mixture_coefficients: chex.Array,
    log_multinomial_probs: chex.Array,
    key: jax.random.KeyArray,
    num_noisy_labels_per_sample: int,
    num_multinomial_samples: int,
    num_classes: int,
    batch_size_em: int,
    num_em_iter: int,
    alpha: float,
    beta: float
) -> tuple[chex.Array, chex.Array]:
    mixture_distribution = tfp.distributions.Categorical(
        logits=log_mixture_coefficients,
        validate_args=True,
        allow_nan_stats=False
    )
    component_distribution = tfp.distributions.Multinomial(
        total_count=num_noisy_labels_per_sample,
        logits=log_multinomial_probs,
        validate_args=True
    )

    mult_mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        components_distribution=component_distribution
    )

    # generate noisy labels
    yhat = mult_mixture.sample(
        sample_shape=(num_multinomial_samples,),
        seed=key
    )  # (sample_shape, N, C)
    yhat = jnp.transpose(a=yhat, axes=(1, 0, 2))  # (N, sample_shape, C)

    log_p_y, log_mult_prob = EM_for_mm(
        y=yhat,
        batch_size_=batch_size_em,
        n_=num_multinomial_samples,
        d_=num_classes,
        num_noisy_labels_per_sample=num_noisy_labels_per_sample,
        num_em_iter=num_em_iter,
        alpha=alpha,
        beta=beta,
        key=key
    )

    return log_p_y, log_mult_prob


def get_p_y(
    log_mixture_coefficients: chex.Array,
    log_multinomial_probs: chex.Array,
    args: argparse.Namespace
) -> tuple[chex.Array, chex.Array]:
    """Infer the noisy label distribution as a C-multinomial mixture model
        using EM. The data is generated from a (K + 1)C-multinomial mixture
        models obtained via nearest-neighbours

    Args:
        log_mixture_coefficients:
        log_multinomial_probs: the probability matrix containing probability
            vectors of (K + 1)C multinomial distributions
        args: object storing configuration parameters

    Returns:
        log_p_y:
        log_mult_prob: a matrix containing the probability vectors of
            C-multinomial distributions
    """
    return _get_p_y(
        log_mixture_coefficients=log_mixture_coefficients,
        log_multinomial_probs=log_multinomial_probs,
        key=args.key,
        num_noisy_labels_per_sample=args.num_noisy_labels_per_sample,
        num_multinomial_samples=args.num_multinomial_samples,
        num_classes=args.num_classes,
        batch_size_em=args.batch_size_em,
        num_em_iter=args.num_em_iter,
        alpha=args.alpha,
        beta=args.beta
    )


@jax.jit
def pred_step(state: train_state.TrainState, x: chex.Array) -> chex.Array:
    logits, _ = state.apply_fn(
        variables={'params': state.params, 'batch_stats': state.batch_stats},
        x=x,
        train=False,
        mutable=['batch_stats']
    )
    return logits


def evaluate(state: train_state.TrainState, ds: tf.data.Dataset) -> chex.Array:
    """Evaluate the model on the given dataset

    Args:
        state: train_state includes params and model
        ds: the batched dataset of interest

    Returns:
        prediction accuracy on the given dataset
    """
    accuracy = metrics.Accuracy(total=jnp.array(0.), count=jnp.array(0))

    for x, y in tqdm(iterable=ds, desc=' validate', position=2, leave=False):
        # move to GPU
        with tf.device('/gpu:0'):
            x = tf.identity(input=x)
            y = tf.identity(input=y)

        # move to DLPack
        x_dl = tf.experimental.dlpack.to_dlpack(tf_tensor=x)
        y_dl = tf.experimental.dlpack.to_dlpack(tf_tensor=y)

        # convert to JAX
        x = jax.dlpack.from_dlpack(x_dl)
        y = jax.dlpack.from_dlpack(y_dl)

        logits = pred_step(state, x)
        accuracy = accuracy.merge(other=metrics.Accuracy.from_model_output(logits=logits, labels=y))

    return accuracy.compute()


@jax.jit
def train_step(state: train_state.TrainState, x: chex.Array, y: chex.Array) -> tuple[train_state.TrainState, chex.Array]:
    """Train for a single step."""
    def loss_fn(params: flax.core.frozen_dict.FrozenDict, batch_stats: flax.core.frozen_dict.FrozenDict) -> tuple[chex.Array, flax.core.frozen_dict.FrozenDict]:
        logits, batch_stats_new = state.apply_fn(
            variables={'params': params, 'batch_stats': batch_stats},
            x=x,
            train=True,
            mutable=['batch_stats']
        )
        loss = optax.softmax_cross_entropy(logits=logits, labels=y).mean()

        return loss, batch_stats_new

    grad_value_fn = jax.value_and_grad(fun=loss_fn, argnums=0, has_aux=True)
    (loss, batch_stats_new), grads = grad_value_fn(state.params, state.batch_stats)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=batch_stats_new['batch_stats'])

    return state, loss


def train_model(state: train_state.TrainState, dataset_train: tf.data.Dataset, p_y: chex.Array, num_epochs: int, aim_run: aim.Run, args: argparse.Namespace) -> train_state.TrainState:
    """
    """
    # generate another dataset for labels
    p_y_dataset = tf.data.Dataset.from_tensor_slices(tensors=p_y)
    assert len(p_y_dataset) == len(dataset_train), \
        f'Dataset lengths mismatch: len(p_y_dataset) = {len(p_y_dataset)}, while len(dataset_train) = {len(dataset_train)}'

    ds_train = tf.data.Dataset.zip(dataset_train, p_y_dataset)
    ds_train = ds_train.map(
        map_func=lambda x, yhat: (tf.cast(x, tf.float32) / 255., yhat)
    )
    ds_train = ds_train.shuffle(buffer_size=args.total_num_samples, reshuffle_each_iteration=True)
    ds_train = ds_train.batch(batch_size=args.batch_size)
    ds_train = ds_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    # data augmentation
    image_augmentation_fn = jax.vmap(
        fun=partial(augment_an_image, image_shape=args.img_shape),
        in_axes=(0, 0)
    )

    # testing dataset
    ds_test, _ = image_folder_label_csv(
        root_dir=os.path.join(args.dataset_root, 'test'),
        csv_path=os.path.join(args.dataset_root, 'split', 'clean_validation'),
        sample_indices=None,
        image_shape=args.img_shape
    )
    ds_test = ds_test.batch(batch_size=args.batch_size)
    ds_test = ds_test.map(
        map_func=lambda x, y: (tf.cast(x, tf.float32) / 255., y)
    )
    ds_test = ds_test.prefetch(buffer_size=tf.data.AUTOTUNE)

    for _ in tqdm(iterable=range(num_epochs), desc='train', leave=False, position=1):

        # define metrics
        loss_accumulate = metrics.Average(total=jnp.array(0.), count=jnp.array(0))

        for x, yhat in tqdm(
            iterable=ds_train,
            desc=' epoch',
            leave=False,
            position=2
        ):
            # move to GPU
            with tf.device('/gpu:0'):
                x = tf.identity(input=x)
                yhat = tf.identity(input=yhat)

            # move to DLPack
            x_dl = tf.experimental.dlpack.to_dlpack(tf_tensor=x)
            yhat_dl = tf.experimental.dlpack.to_dlpack(tf_tensor=yhat)

            # Load arrays from the DLPack
            x = jax.dlpack.from_dlpack(x_dl)
            yhat = jax.dlpack.from_dlpack(yhat_dl)

            # data augmentation
            args.key = jax.random.split(key=args.key, num=1).squeeze()
            keys = jax.random.split(key=args.key, num=x.shape[0])
            x = image_augmentation_fn(keys, x)

            state, loss = train_step(state, x, yhat)

            loss_accumulate = metrics.Average.merge(self=loss_accumulate, other=metrics.Average.from_model_output(values=loss))

        aim_run.track(
            value=loss_accumulate.compute(),
            name='Loss',
            context={'model': int(state.model_id)}
        )

        # region EVALUATION
        acc = evaluate(state=state, ds=ds_test)
        aim_run.track(value=acc, name='Accuracy', context={'model': int(state.model_id)})
        # endregion

    return state


class TrainState(train_state.TrainState):
    """A data-class storing model's parameters, optimizer and others
    """
    batch_stats: Any
    model_id: chex.Numeric


class AddNoiseState(NamedTuple):
    """State for adding gradient noise. Contains a count for annealing."""
    count: chex.Array
    rng_key: chex.PRNGKey


def add_Langevin_dynamics_noise(
    lr_schedule_fn: Callable,
    len_dataset: int,
    seed: int
) -> optax._src.base.GradientTransformation:
    """Adopted from optax.add_noise. This is to perform stochastic gradient
    Langevin dynamics.
    """

    def init_fn(params):
        del params
        return AddNoiseState(
            count=jnp.zeros([], jnp.int32),
            rng_key=jax.random.PRNGKey(seed)
        )

    def update_fn(updates, state, params=None):
        del params
        num_vars = len(jax.tree_util.tree_leaves(updates))
        treedef = jax.tree_util.tree_structure(updates)
        count_inc = optax._src.numerics.safe_int32_increment(state.count)
        variance = 2 * lr_schedule_fn(count_inc)
        standard_deviation = jnp.sqrt(variance) / len_dataset
        all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
        noise = jax.tree_util.tree_map(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
            updates, jax.tree_util.tree_unflatten(treedef, all_keys[1:])
        )
        updates = jax.tree_util.tree_map(
            lambda g, n: g + standard_deviation.astype(g.dtype) * n,
            updates, noise)
        return updates, AddNoiseState(count=count_inc, rng_key=all_keys[0])

    return optax._src.base.GradientTransformation(init_fn, update_fn)
