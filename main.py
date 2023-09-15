import jax
from jax import numpy as jnp
import jax.dlpack  # transfer between jax and tf

import flax
from flax import linen as nn
from flax.training import train_state
import optax

from jaxopt import OSQP

import dm_pix  # deep-mind image processing

from clu import metrics

import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow_probability.substrates import jax as tfp

import numpy as np

import faiss  # facebook similarity search

import aim

import random

from functools import partial

import subprocess
import os
from pathlib import Path

from tqdm import tqdm
import logging
import argparse
from typing import Any
import chex

from PreactResnet import ResNet18
from AnnotatedDataset import ImageFolder_2
from utils import EM_for_mm


def parse_arguments() -> argparse.Namespace:
    """
    """
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--experiment-name', type=str, help='')

    parser.add_argument('--dataset-root', type=str, default=None, help='Path to the folder containing train and test sets')
    # parser.add_argument('--csv-file-path', type=str, default=None)
    # parser.add_argument('--delimiter', type=str, default=' ')
    parser.add_argument('--logdir', type=str, default='logs', help='Folder to store logs')

    parser.add_argument('--img-shape', action='append', help='e.g., 32 32 3 or 224 224 3')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples in each iteration. None is whole dataset')

    # parser.add_argument('--num-classes', type=int, help='Number of classes')
    parser.add_argument('--noise-rate', type=float, help='Noise rate')

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
    # parser.add_argument('--p-dropout', type=float, default=0., help='Dropout')
    # endregion

    # parser.add_argument('--num-adapt-epochs', type=int, default=1, help='Number of epochs to train model after every EM')

    parser.add_argument('--num-samples-monitor', type=int, default=None)

    parser.add_argument('--network-architecture', type=str, default='PreActResNet', help='PreActResNet or ResNet')

    parser.add_argument('--ssl', dest='ssl', action='store_true')
    parser.add_argument('--no-ssl', dest='ssl', action='store_false')
    parser.set_defaults(ssl=False)

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument('--jax-mem-fraction', type=float, default=0.1, help='Percentage of GPU memory allocated for Jax')

    args = parser.parse_args()

    return args


def get_features(state: train_state.TrainState, split_name: str, sample_indices: chex.Array, args: argparse.Namespace) -> chex.Array:
    """Extract features of data

    Args:
        state: a data-class storing model-related parameters
        split_name: either 'train' or 'test'
        sample_indices: the indices of the samples being feature-extracted
        args: configuration object

    Returns:
        xb: the extracted features
    """
    # setup a function to extract features
    def feature_fn(
        model: nn.Module,
        params: flax.core.frozen_dict.FrozenDict,
        x: chex.Array
    ):
        return model.apply(
            variables=params,
            x=x,
            mutable=['batch_stats'],
            method=lambda module, x: module.features(x=x, train=False)
        )

    ds_builder = ds_builder = ImageFolder_2(
        root_dir=args.dataset_root,
        noise_rate=args.noise_rate,
        split_name=split_name,
        sample_indices=sample_indices,
        shape=args.img_shape
    )
    data_loader = ds_builder.as_dataset(
        split='train',
        shuffle_files=False,
        as_supervised=True,
        batch_size=args.batch_size
    )

    # get feature dimension
    for x, _ in tqdm(iterable=tfds.as_numpy(dataset=data_loader), desc=' Extract features', leave=False, position=1):
        x = jnp.array(object=x) / 255.
        features, _ = feature_fn(
            model=state.model,
            params={'params': state.params, 'batch_stats': state.batch_stats},
            x=x
        )
        break

    xb = jnp.empty(shape=(sample_indices.size, features.shape[-1]))  # (N, D)

    # region EXTRACT FEATURES
    start_idx = 0
    for x, _ in tqdm(
        iterable=data_loader,
        desc='Extract features',
        leave=False,
        position=1
    ):
        x = jnp.array(object=x) / 255.
        features, _ = feature_fn(
            model=state.model,
            params={'params': state.params, 'batch_stats': state.batch_stats},
            x=x
        )

        end_idx = start_idx + x.shape[0]
        # xb[start_idx:end_idx, :] = features
        xb = xb.at[start_idx:end_idx].set(features)
        start_idx = end_idx
    # endregion

    return xb


def get_knn_indices(xb: np.ndarray, sample_indices: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """find the indices of k-nearest neighbours

    Args:
        xb: the matrix where each row contains feature vector of one datum
        sample_indices: the original indices of the samples (w.r.t. whole dataset)
        args:

    Returns:
        index_matrix: matrix containing indices of nearest neighbours
    """
    res = faiss.StandardGpuResources()  # use a single GPU

    index_flat = faiss.IndexFlatL2(xb.shape[-1])  # build a flat (CPU) index
    index_id_map = faiss.IndexIDMap(index_flat)  # translates ids when adding and searching
    gpu_index_flat = faiss.index_cpu_to_gpu(provider=res, device=0, index=index_id_map)  # make it into a gpu index

    gpu_index_flat.add_with_ids(xb, sample_indices)  # add vectors to the index

    # start the nearest-neighbour search
    _, index_matrix = gpu_index_flat.search(x=xb, k=args.k + 1)  # adding 1 is because the return includes the sample itself

    return index_matrix[:, 1:]  # exclude the first element which is the sample itself


@jax.jit
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
    x = jnp.abs(x)  # non-negative number
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


def get_p_y(
        log_mixture_coefficients: chex.Array,
        log_multinomial_probs: chex.Array,
        args: argparse.Namespace) -> tuple[chex.Array, chex.Array]:
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
    mixture_distribution = tfp.distributions.Categorical(
        logits=log_mixture_coefficients,
        validate_args=True,
        allow_nan_stats=False
    )
    component_distribution = tfp.distributions.Multinomial(
        total_count=args.num_noisy_labels_per_sample,
        logits=log_multinomial_probs,
        validate_args=True
    )

    mult_mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        components_distribution=component_distribution
    )

    # generate noisy labels
    rng = np.random.default_rng(seed=None)
    yhat = mult_mixture.sample(
        sample_shape=(args.num_multinomial_samples,),
        seed=jax.random.PRNGKey(seed=rng.integers(low=0, high=2**63))
    )  # (sample_shape, N, C)
    yhat = jnp.transpose(a=yhat, axes=(1, 0, 2))  # (N, sample_shape, C)

    log_p_y, log_mult_prob = EM_for_mm(y=yhat, args=args)

    return log_p_y, log_mult_prob


def relabel_data(
        log_p_y: chex.Array,
        log_mult_prob: chex.Array,
        nn_idx: chex.Array,
        coding_matrix: chex.Array,
        sample_indices: chex.Array,
        args: argparse.Namespace) -> tuple[chex.Array, chex.Array]:
    """Perform EM to infer the p(y | x) and p(yhat | x, y)
    In this implementation, both p(y | x) and p(yhat | x, y) are sparse tensors
    and assumed that their dense dimensions are the same across all samples.
    This assumption enables the vectorization in the implementation.

    Args:
        log_p_y: a row-wise matrix containing the "cleaner" label distributions of samples  (N , C)
        log_mult_prob: a tensor containing the transition matrices of samples  (N, C, C)
        nn_idx: a matrix where each row contains the indices of nearest-neighbours corresponding to row (sample) id  (N, K)
        coding_matrix: a matrix where each row contains the coding coefficient (normalised similarity)  (N, K)
        args: contains configuration parameters
        sample_indices: a 1-d vector (array) containing indices of samples considered

    Returns:
        p_y: a sparse matrix where each row is p(y | x)  # (N, C)
        mult_prob: a sparse 3-d tensor where each matrix is p(yhat | x, y)  # (N, C , C)
    """
    # initialize new p(y | x) and p(yhat | x, y)
    log_p_y_new = log_p_y + 0.
    log_mult_prob_new = log_mult_prob + 0.

    data_loader = tf.data.Dataset.from_tensor_slices(tensors=sample_indices)
    data_loader = data_loader.batch(batch_size=args.batch_size_em)

    current_index = 0
    for indices in tqdm(iterable=tfds.as_numpy(dataset=data_loader), desc=' re-label', leave=False, position=1):
        # track sample size
        batch_size_ = indices.size

        # extract the corresponding noisy labels
        log_p_y_ = log_p_y[indices]  # (B, C)
        log_mult_prob_ = log_mult_prob[indices]  # (B, C, C)

        # extract coding coefficients
        log_nn_coding = jnp.log(coding_matrix[indices])  # (B, K)

        # region APPROXIMATE p(y | x) through nearest-neighbours
        nn_idx_ = nn_idx[current_index:(current_index + batch_size_)]
        log_p_y_nn = log_p_y[nn_idx_]  # (B, K, C)
        log_mult_prob_nn = log_mult_prob[nn_idx_]  # (B, K, C, C)
        log_mult_prob_nn = jnp.reshape(a=log_mult_prob_nn, newshape=(log_mult_prob_nn.shape[0], -1, log_mult_prob_nn.shape[-1]))  # (B, K*C, C)

        # calculate the mixture coefficients of other mixtures
        # induced by nearest-neighbours
        log_mixture_coefficient_nn = jnp.log(1 - args.mu) + log_nn_coding[:, :, None] + log_p_y_nn  # (B, K, C)
        log_mixture_coefficient_nn = jnp.reshape(a=log_mixture_coefficient_nn, newshape=(log_mixture_coefficient_nn.shape[0], -1))  # (B, K*C)

        log_mixture_coefficient_x = jnp.log(args.mu) + log_p_y_  # (B, C)

        log_mixture_coefficient = jnp.concatenate(arrays=(log_mixture_coefficient_x, log_mixture_coefficient_nn), axis=-1)  # (B, (K + 1) * C)
        log_multinomial_probs = jnp.concatenate(arrays=(log_mult_prob_, log_mult_prob_nn), axis=1)  # (B, (K + 1) * C, C)
        # endregion

        # predict the clean label distribution using EM
        log_p_y_temp, log_mult_prob_temp = get_p_y(
            log_mixture_coefficients=log_mixture_coefficient,
            log_multinomial_probs=log_multinomial_probs,
            args=args
        )

        if jnp.any(a=jnp.isnan(log_p_y_temp)):
            raise ValueError('NaN is detected after running EM')

        # update the noisy labels
        log_p_y_new = log_p_y_new.at[indices].set(values=log_p_y_temp)
        log_mult_prob_new = log_mult_prob_new.at[indices].set(values=log_mult_prob_temp)

        # update current index
        current_index = current_index + batch_size_

    return log_p_y_new, log_mult_prob_new


def cross_entropy_loss(
    params: flax.core.frozen_dict.FrozenDict,
    batch_stats: flax.core.frozen_dict.FrozenDict,
    model: nn.Module,
    x: jax.Array,
    y: jax.Array,
    train: bool
) -> tuple[jax.Array, jax.Array]:
    """
    """
    logit, updates = model.apply(
        variables={'params': params, 'batch_stats': batch_stats},
        x=x,
        train=train,
        mutable=['batch_stats']
    )  # return a tuple(loss, mutable vars)

    loss = optax.softmax_cross_entropy(logits=logit, labels=y)

    loss = jnp.mean(a=loss, axis=0)

    return loss, updates


def evaluate(state: train_state.TrainState, args: argparse.Namespace) -> chex.Array:
    """
    """
    builder = tfds.ImageFolder(root_dir=args.dataset_root)
    split_name = 'test'
    ds = builder.as_dataset(split=split_name, shuffle_files=False, as_supervised=True, batch_size=args.batch_size)

    accuracy = metrics.Accuracy(total=jnp.array(0.), count=jnp.array(0))

    for image, label in tqdm(iterable=tfds.as_numpy(dataset=ds), desc=' validate', position=2, leave=False):
        x = jnp.array(object=image) / 255.
        y = jnp.array(label)

        logits, batch_stats = state.model.apply(variables={'params': state.params, 'batch_stats': state.batch_stats}, x=x, train=False, mutable=['batch_stats'])
        accuracy = accuracy.merge(other=metrics.Accuracy.from_model_output(logits=logits, labels=y))

    return accuracy.compute()


def train_model(state: train_state.TrainState, dataset_train: tf.data.Dataset, p_y: chex.Array, num_epochs: int, aim_run: aim.Run, args: argparse.Namespace) -> train_state.TrainState:
    """
    """
    # define loss function
    loss_grad_fn = partial(cross_entropy_loss, model=state.model, train=True)
    loss_grad_fn = jax.value_and_grad(fun=loss_grad_fn, argnums=0, has_aux=True)
    loss_grad_fn = jax.jit(fun=loss_grad_fn)

    # generate another dataset for labels
    p_y_dataset = tf.data.Dataset.from_tensor_slices(tensors=p_y)
    dataset_noisy_labels = tf.data.Dataset.zip(dataset_train, p_y_dataset)
    dataset_noisy_labels = dataset_noisy_labels.shuffle(buffer_size=args.total_num_samples, reshuffle_each_iteration=True)
    dataset_noisy_labels = dataset_noisy_labels.batch(batch_size=args.batch_size)
    dataset_noisy_labels = dataset_noisy_labels.prefetch(tf.data.AUTOTUNE)

    # numpy random generator
    rng = np.random.default_rng(seed=None)
    rand_flip_ud = jax.vmap(fun=dm_pix.random_flip_up_down, in_axes=(0, 0))
    rand_flip_lr = jax.vmap(fun=dm_pix.random_flip_left_right, in_axes=(0, 0))
    augmentation_list = [rand_flip_ud, rand_flip_lr]

    for _ in tqdm(iterable=range(num_epochs), desc='train', leave=False, position=1):

        # define metrics
        loss_accumulate = metrics.Average(total=jnp.array(0.), count=jnp.array(0))

        for (x, y), yhat in tqdm(iterable=tfds.as_numpy(dataset=dataset_noisy_labels), desc=' epoch', leave=False, position=2):
            x = jnp.array(object=x, dtype=jnp.float32) / 255.
            yhat = jnp.array(object=yhat)

            # data augmentation
            key = jax.vmap(fun=jax.random.PRNGKey)(rng.integers(low=0, high=2**63, size=y.size))
            augment_fun = random.sample(population=augmentation_list, k=1)[0]
            x = augment_fun(key, x)

            (loss, batch_stats_new), grads = loss_grad_fn(state.params, batch_stats=state.batch_stats, x=x, y=yhat)

            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=batch_stats_new['batch_stats'])

            loss_accumulate = metrics.Average.merge(self=loss_accumulate, other=metrics.Average.from_model_output(values=loss))

        aim_run.track(value=loss_accumulate.compute(), name='Loss', context={'model': state.model_id})

        # region EVALUATION
        acc = evaluate(state=state, args=args)
        aim_run.track(value=acc, name='Accuracy', context={'model': state.model_id})
        # endregion

    return state


class TrainState(train_state.TrainState):
    """A data-class storing model's parameters, optimizer and others
    """
    batch_stats: Any
    model: nn.Module
    model_id: int


def main() -> None:
    """Main function
    """
    # parse input arguments
    args = parse_arguments()
    # print(xla_bridge.get_backend('cpu'))

    # region JAX CONFIGURATION
    # set jax memory allocation
    assert args.jax_mem_fraction < 1. and args.jax_mem_fraction > 0.
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.jax_mem_fraction)

    # configure CUDA for JAX
    jax.config.update('jax_platforms', 'cuda')

    # allocate TensorFlow GPU memory
    # tf.config.experimental.set_visible_devices([], 'GPU')  # disable GPU

    # limit GPU memory for TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        device=gpus[0],
        logical_devices=[tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
    )
    # endregion

    # region DATASET
    # convert image-shape from string to int
    args.img_shape = [int(ishape) for ishape in args.img_shape]
    ds_builder = ImageFolder_2(root_dir=args.dataset_root, noise_rate=args.noise_rate, split_name='train', shape=args.img_shape)

    args.num_classes = len(ds_builder.info.features['label'].names)
    args.total_num_samples = ds_builder.info.splits['train'].num_examples

    # get noisy labels
    dataset_train = ds_builder.as_dataset(split='train', shuffle_files=False, as_supervised=True, batch_size=args.batch_size)
    log_p_y_1 = jnp.empty(shape=(args.total_num_samples, args.num_classes), dtype=jnp.float32)  # (N, C)
    current_index = 0
    for x, yhat in tqdm(iterable=tfds.as_numpy(dataset=dataset_train), desc='get noisy labels', leave=False, position=1):
        # track how many samples
        batch_size_ = yhat.shape[0]

        # convert integer numbers to one-hot vectors
        noisy_labels = jax.nn.one_hot(x=yhat, num_classes=args.num_classes, dtype=jnp.float32)

        # assign to yhats
        log_p_y_1 = log_p_y_1.at[current_index:(current_index + batch_size_)].set(noisy_labels)

        current_index = current_index + batch_size_

    # initialize p_y and mult_probs
    log_p_y_1 = optax.smooth_labels(labels=log_p_y_1, alpha=0.1)
    log_p_y_1 = jnp.log(log_p_y_1)

    log_mult_prob_1 = jnp.eye(N=args.num_classes)  # (C, C)
    log_mult_prob_1 = jnp.tile(A=log_mult_prob_1[None, :, :], reps=(args.total_num_samples, 1, 1))  # (N, C, C)

    # add random noise
    log_mult_prob_1 = 10 * log_mult_prob_1 + np.random.rand(*log_mult_prob_1.shape)
    log_mult_prob_1 = jax.nn.log_softmax(x=log_mult_prob_1, axis=-1)

    log_p_y_2 = log_p_y_1 + 0.
    log_mult_prob_2 = log_mult_prob_1 + 0.

    del current_index
    del batch_size_

    dataset_train = ds_builder.as_dataset(split='train', as_supervised=True)
    # endregion

    # region MODEL
    model = ResNet18(num_classes=len(ds_builder.info.features['label'].names))

    lr_schedule_fn = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=500*args.num_epochs
    )

    def initialize_train_state(model_id: int) -> train_state.TrainState:
        key1, key2 = jax.random.split(
            key=jax.random.PRNGKey(random.randint(a=0, b=1_000)),
            num=2
        )
        x = jax.random.normal(key1, (1, 32, 32, 3))  # Dummy input data
        params = model.init(rngs=key2, x=x, train=False)
        tx = optax.sgd(learning_rate=lr_schedule_fn, momentum=0.9)  # define an optimizer

        state = TrainState.create(
            apply_fn=model.apply,
            params=params['params'],
            batch_stats=params['batch_stats'],
            model=model,
            model_id=model_id,
            tx=tx
        )

        return state

    state_1 = initialize_train_state(model_id=1)
    state_2 = initialize_train_state(model_id=2)
    # endregion

    # region EXPERIMENT TRACKING
    # create log folder if it does not exist
    logging.info(msg='Initialise AIM repository to store logs')
    if not os.path.exists(path=args.logdir):
        logging.info(msg='Logging folder not found. Make a logdir at {0:s}'.format(args.logdir))
        Path(args.logdir).mkdir(parents=True, exist_ok=True)

    if not aim.sdk.repo.Repo.exists(path=args.logdir):
        logging.info(msg='Initialize AIM repository')
        # aim.sdk.repo.Repo(path=args.logdir, read_only=False, init=True)
        subprocess.run(args=["aim", "init"])

    aim_run = aim.Run(
        repo=args.logdir,
        read_only=False,
        experiment=args.experiment_name,
        force_resume=False,
        capture_terminal_logs=False,
        system_tracking_interval=600  # capture every x seconds
    )
    # endregion

    # # region WARM-UP
    for state_ in [state_1, state_2]:
        state_ = train_model(
            state=state_,
            dataset_train=dataset_train,
            p_y=jnp.exp(log_p_y_1),
            num_epochs=args.num_warmup,
            aim_run=aim_run,
            args=args
        )
    # # endregion

    # define data-loaders to create sample indices
    def create_index_loader() -> tf.data.Dataset:
        index_loader = tf.data.Dataset.range(args.total_num_samples)
        index_loader = index_loader.shuffle(buffer_size=args.total_num_samples, reshuffle_each_iteration=True)
        index_loader = index_loader.batch(batch_size=args.num_samples)

        return index_loader

    index_loader_1 = create_index_loader()
    index_loader_2 = create_index_loader()

    try:
        for epoch_id in tqdm(iterable=range(args.num_epochs), desc='Epoch', position=0):

            for sample_indices_1, sample_indices_2 in zip(tfds.as_numpy(dataset=index_loader_1), tfds.as_numpy(dataset=index_loader_2)):

                def relabel_data_wrapper(
                        state_: train_state.TrainState,
                        sample_indices_: chex.Array,
                        log_p_y_: chex.Array,
                        log_mult_prob_: chex.Array
                ) -> tuple[chex.Array, chex.Array]:
                    """This is a wrapper to execute the following actions:
                    - extract features
                    - find nearest neighbours
                    - solve for similarity coding coefficients
                    - re-label data
                    """
                    # extract features
                    features = get_features(state=state_, split_name='train', sample_indices=sample_indices_, args=args)
                    features = np.asanyarray(a=features)

                    # find K nearest-neighbours
                    knn_indices_ = get_knn_indices(xb=features, sample_indices=sample_indices_, args=args)

                    # calculate coding mtrices
                    coding_matrix_ = get_batch_local_affine_coding(samples=features, knn_indices=knn_indices_)

                    # run EM and re-label samples
                    return relabel_data(
                        log_p_y=log_p_y_,
                        log_mult_prob=log_mult_prob_,
                        nn_idx=knn_indices_,
                        coding_matrix=coding_matrix_,
                        sample_indices=sample_indices_,
                        args=args
                    )

                log_p_y_1, log_mult_prob_1 = relabel_data_wrapper(
                    state_=state_1,
                    sample_indices_=sample_indices_1,
                    log_p_y_=log_p_y_1,
                    log_mult_prob_=log_mult_prob_1
                )

                log_p_y_2, log_mult_prob_2 = relabel_data_wrapper(
                    state_=state_2,
                    sample_indices_=sample_indices_2,
                    log_p_y_=log_p_y_2,
                    log_mult_prob_=log_mult_prob_2
                )

            # region TRAIN models on relabelled data
            state_1 = train_model(
                state=state_1,
                dataset_train=dataset_train,
                p_y=jnp.exp(log_p_y_2),
                num_epochs=1,
                aim_run=aim_run,
                args=args
            )
            state_2 = train_model(
                state=state_2,
                dataset_train=dataset_train,
                p_y=jnp.exp(log_p_y_1),
                num_epochs=1,
                aim_run=aim_run,
                args=args
            )
            # endregion

        logging.info(msg='Training is completed.')
    finally:
        aim_run.close()
        logging.info(msg='AIM is closed.\nProgram is terminated.')


if __name__ == '__main__':
    logger_current = logging.getLogger(name=__name__)
    logger_current.setLevel(level=logging.INFO)
    main()
