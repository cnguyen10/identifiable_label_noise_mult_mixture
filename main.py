import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jax.sharding import PartitionSpec, NamedSharding

import flax.nnx as nnx
from flax.traverse_util import flatten_dict

import orbax.checkpoint as ocp

import optax

from tensorflow_probability.substrates import jax as tfp

import grain.python as grain

import mlflow

import random

from functools import partial

import os
from pathlib import Path

from tqdm import tqdm
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from typing import Any

from DataSource import ImageDataSource

from utils import get_knn_indices, get_batch_local_affine_coding, EM_for_mm

from utils import init_tx, initialize_dataloader
from mixup import mixup_data

from models.ResNet import ResNet, PreActResNet


class FeatureEmbeddedModel(nnx.Module):
    def __init__(
        self,
        feature_extractor: ResNet | PreActResNet,
        num_classes: int,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32
    ) -> None:
        super().__init__()

        self.feature_extractor = feature_extractor

        self.clf = nnx.Linear(
            in_features=feature_extractor.clf.out_features,
            out_features=num_classes,
            dtype=dtype,
            rngs=rngs
        )

    def get_features(self, x: jax.Array) -> jax.Array:
        """
        """
        return self.feature_extractor(x)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = self.get_features(x=x)
        out = self.clf(out)

        return out


def get_data_sources(
    train_file: str,
    test_file: str,
    root: str
) -> tuple[grain.RandomAccessDataSource, grain.RandomAccessDataSource]:
    """load the data sources

    Args:
        train_file: path to the training json file
        test_file: path to the testing json file
        root: the root path to concatenate to the path of each sample in
    those json files to load data

    Returns:
        source_train: data source for training
        source_test: data source for testing
    """
    source_train = ImageDataSource(
        annotation_file=train_file,
        root=root,
        idx_list=None
    )

    source_test = ImageDataSource(
        annotation_file=test_file,
        root=root,
        idx_list=None
    )

    return (source_train, source_test)


def extract_features(
    model: nnx.Module,
    dataloader: grain.DataLoader,
    cfg: DictConfig
) -> jax.Array:
    """
    """
    model.eval()

    features = []
    for samples in dataloader:
        x = jnp.array(object=samples['image'], dtype=eval(cfg.jax.dtype))

        features_temp = model.get_features(
            x=x
        )  # pyright: ignore[reportAttributeAccessIssue]

        features.append(features_temp)

    features = jnp.concatenate(arrays=features, axis=0)
    return features


def get_soft_noisy_labels(
    data_source: grain.RandomAccessDataSource,
    num_classes: int,
    seed: int | None = None,
    progress_bar_flag: bool = True
) -> dict[str, jax.Array]:
    """aggregate multiple noisy labels
    """
    key = jax.random.key(
        seed=seed if seed is not None else random.randint(a=0, b=1_000)
    )

    p_y = jnp.zeros(
        shape=(len(data_source), num_classes),
        dtype=jnp.float32
    )
    for i in tqdm(
        iterable=range(len(data_source)),
        desc='Load labels',
        ncols=80,
        leave=False,
        position=0,
        colour='green',
        disable=not progress_bar_flag
    ):
        noisy_labels = data_source[i]['label']  # list[int]
        noisy_labels = jnp.array(object=noisy_labels, dtype=jnp.int32)

        noisy_labels_onehot = jax.nn.one_hot(
            x=noisy_labels,
            num_classes=num_classes
        )

        if noisy_labels_onehot.ndim == 1:
            noisy_labels_aggregate = noisy_labels_onehot
        elif noisy_labels_onehot.ndim == 2:
            noisy_labels_aggregate = jnp.mean(a=noisy_labels_onehot, axis=0)
        else:
            raise ValueError(' '.join((
                'Dimensions of labels are incorrect.',
                'It only accept either an int scalar or a list of int'
            )))

        p_y = p_y.at[i].set(values=noisy_labels_aggregate)

    p_y = optax.smooth_labels(labels=p_y, alpha=0.1)

    log_mult_prob = jnp.eye(N=num_classes)  # (C, C)
    log_mult_prob = jnp.tile(
        A=log_mult_prob[None, :, :],
        reps=(len(data_source), 1, 1)
    )  # (N, C, C)

    # add random noise
    log_mult_prob = 10 * log_mult_prob \
        + jax.random.uniform(key=key, shape=log_mult_prob.shape)
    log_mult_prob = jax.nn.log_softmax(x=log_mult_prob, axis=-1)

    return dict(p_y=p_y, log_mult_prob=log_mult_prob)


def get_sparse_noisy_labels(
    data_source: grain.RandomAccessDataSource,
    num_classes: int,
    progress_bar_flag: bool = False
) -> dict[str, Any]:
    """load the noisy labels as multinomial mixtures in the sparse setting

    Args:
        data_source:
        num_classes: the total number of classes
        num_approx_classes: the actual number of classes (non-zero elements)

    Return:
        mult_mixture: a dictionary contains:
            - p_y: mixture coefficient or clean label dist.
            - log_mult_prob:
    """
    # initialise parameters of multinomial mixture model
    log_mult_prob = jnp.empty(
        shape=(len(data_source), num_classes),
        dtype=jnp.float32
    )

    p_y = dict(
        data=jnp.ones(shape=(len(data_source), 1)),
        indices=jnp.empty(
            shape=(len(data_source), 1),
            dtype=jnp.int32
        )
    )

    data_loader = (
        grain.MapDataset.source(source=data_source)
        .shuffle(seed=0)  # Shuffles globally.
        .map(lambda x: (x['idx'], x['label']))  # Maps each element.
        .batch(
            batch_size=jnp.gcd(x1=len(data_source), x2=200).item()
        )  # Batches consecutive elements.
    )

    for indices, labels in tqdm(
        iterable=data_loader,
        desc='load noisy labels',
        leave=False,
        ncols=80,
        position=0,
        colour='green',
        disable=not progress_bar_flag
    ):
        # store the indices (or class id)
        p_y['indices'] = (
            p_y['indices']
            .at[indices]
            .set(values=labels[:, None])
        )

        # log of multinomial probability vector
        log_mult_prob_temp = jnp.log(optax.smooth_labels(
            labels=jax.nn.one_hot(
                x=labels,
                num_classes=num_classes,
                dtype=jnp.float32
            ),
            alpha=0.1
        ))
        log_mult_prob = (
            log_mult_prob
            .at[indices]
            .set(values=log_mult_prob_temp)
        )

    return dict(p_y=p_y, log_mult_prob=log_mult_prob[:, None, :])


def p_y_sparse_to_dense(
        data: jax.Array,  # (N, C0)
        indices: jax.Array,  # (N, C0)
        num_classes: int) -> jax.Array:
    """
    """
    # construct the indices for BCOO
    cols = jnp.expand_dims(a=indices, axis=-1)  # (N, C0, 1)
    rows = jnp.broadcast_to(
        array=jnp.expand_dims(a=jnp.arange(len(data)), axis=(-1, -2)),
        shape=cols.shape
    )
    bcoo_indices = jnp.concatenate(arrays=(rows, cols), axis=-1)  # (N, C0, 2)
    bcoo_indices = jnp.reshape(a=bcoo_indices, shape=(-1, 2))

    p_y_sparse = sparse.BCOO(
        args=(data.flatten(), bcoo_indices),
        shape=(len(data), num_classes)
    )

    return sparse.bcoo_todense(mat=p_y_sparse)


@nnx.jit
def cross_entropy_loss(
    model: nnx.Module,
    x: jax.Array,
    y: jax.Array
) -> jax.Array:
    """
    """
    logits = model(x)  # pyright: ignore[reportCallIssue]

    loss = optax.losses.softmax_cross_entropy(
        logits=logits,
        labels=y
    )

    loss = jnp.mean(a=loss, axis=0)

    return loss


@nnx.jit
def train_step(
    x: jax.Array,
    y: jax.Array,
    model: nnx.Module,
    optimizer: nnx.Optimizer
) -> tuple[nnx.Module, nnx.Optimizer, jax.Array]:
    """
    """
    grad_value_fn = nnx.value_and_grad(f=cross_entropy_loss, argnums=0)
    loss, grads = grad_value_fn(model, x, y)

    optimizer.update(model=model, grads=grads)

    return (model, optimizer, loss)


def train_cross_entropy(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    dataloader: grain.DatasetIterator,
    p_y: jax.Array | dict[str, jax.Array],
    cfg: DictConfig
) -> tuple[nnx.Module, nnx.Optimizer, jax.Array]:
    """train a model with cross-entropy loss
    """
    # metric to track the training loss
    loss_accum = nnx.metrics.Average()

    # set train mode
    model.train()

    for _ in tqdm(
        iterable=range(cfg.dataset.length.train // cfg.training.batch_size),
        desc='train',
        ncols=80,
        leave=False,
        position=1,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        samples = next(dataloader)

        x = jnp.array(object=samples['image'], dtype=eval(cfg.jax.dtype))

        if isinstance(p_y, dict):
            y = p_y_sparse_to_dense(
                data=p_y['data'][samples['idx']],
                indices=p_y['indices'][samples['idx']],
                num_classes=cfg.dataset.num_classes
            )
        else:
            y = p_y[samples['idx']]

        if cfg.mixup.enable:
            key = jax.random.PRNGKey(seed=optimizer.step.value)
            x_mixed, y = mixup_data(
                x=x,
                y=y,
                key=key,
                beta_a=cfg.mixup.beta.a,
                beta_b=cfg.mixup.beta.b
            )
            x = jnp.astype(x_mixed, x.dtype)

        model, optimizer, loss = train_step(
            x=x,
            y=y,
            model=model,
            optimizer=optimizer
        )

        if jnp.isnan(loss):
            raise ValueError('Training loss is NaN.')

        loss_accum.update(values=loss)

    return (model, optimizer, loss_accum.compute())


def evaluate(
    model: nnx.Module,
    dataloader: grain.DataLoader,
    cfg: DictConfig
) -> jax.Array:
    """
    """
    model.eval()

    acc_accum = nnx.metrics.Accuracy()

    for samples in tqdm(
        iterable=dataloader,
        total=cfg.dataset.length.test // cfg.training.batch_size + 1,
        desc='eval',
        ncols=80,
        leave=False,
        position=1,
        colour='blue',
        disable=not cfg.data_loading.progress_bar
    ):
        x = jnp.array(object=samples['image'], dtype=eval(cfg.jax.dtype))
        y = jnp.array(object=samples['label'], dtype=jnp.int32)

        logits = model(x)

        acc_accum.update(logits=logits, labels=y)

    return acc_accum.compute()



def get_p_y(
    log_mixture_coefficients: jax.Array,
    log_multinomial_probs: jax.Array,
    key: jax._src.prng.PRNGKeyArray,
    num_noisy_labels_per_sample: int,
    num_multinomial_samples: int,
    num_classes: int,
    num_em_iter: int,
    alpha: float,
    beta: float
) -> tuple[jax.Array, jax.Array]:
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

    log_p_y, log_mult_prob = EM_for_mm(
        approx_mm=mult_mixture,
        n_=num_multinomial_samples,
        d_=num_classes,
        num_noisy_labels_per_sample=num_noisy_labels_per_sample,
        num_em_iter=num_em_iter,
        alpha=alpha,
        beta=beta,
        key=key
    )

    return log_p_y, log_mult_prob


def relabel_data(
    mult_mixture: dict[str, jax.Array],
    nn_idx: jax.Array,
    coding_matrix: jax.Array,
    cfg: DictConfig
) -> dict[str, jax.Array]:
    """Perform EM to infer the p(y | x) and p(yhat | x, y)
    In this implementation, both p(y | x) and p(yhat | x, y) are sparse tensors
    and assumed that their dense dimensions are the same across all samples.
    This assumption enables the vectorization in the implementation.

    Args:
        mult_mixture: a dictionary consisting of two keys:
            log_p_y: a row-wise matrix containing the "cleaner" label
                distributions of samples  (N , C)
            log_mult_prob: a tensor containing the transition matrices of
                samples  (N, C, C)
        nn_idx: a matrix where each row contains the indices of
            nearest-neighbours corresponding to row (sample) id  (N, K)
        coding_matrix: a matrix where each row contains the coding coefficient
            (normalised similarity)  (N, K)
        cfg: contains configuration parameters

    Returns:
        a dictionary consists of two keys:
            log_p_y: a sparse matrix where each row is p(y | x)  # (N, C)
            log_mult_prob: a sparse 3-d tensor where each matrix
                is p(yhat | x, y)  # (N, C , C)
    """
    num_devices = jax.device_count(backend='gpu')

    log_p_y = jnp.log(mult_mixture['p_y'])
    log_mult_prob = mult_mixture['log_mult_prob']

    # initialize new p(y | x) and p(yhat | x, y)
    log_p_y_new = log_p_y + 0.
    log_mult_prob_new = log_mult_prob + 0.

    seed = random.randint(a=0, b=100_000)
    data_loader = (
        grain.MapDataset.range(start=0, stop=len(log_p_y), step=1)
        .shuffle(seed=seed)  # Shuffles globally.
        .map(lambda x: x)  # Maps each element.
        .batch(
            batch_size=cfg.hparams.batch_size_em
        )  # Batches consecutive elements.
    )

    key = jax.random.key(seed=seed)

    get_p_y_fn = partial(
        get_p_y,
        num_noisy_labels_per_sample=cfg.hparams.num_noisy_labels_per_sample,
        num_multinomial_samples=cfg.hparams.num_multinomial_samples,
        num_classes=cfg.dataset.num_classes,
        num_em_iter=cfg.hparams.num_em_iter,
        alpha=cfg.hparams.alpha,
        beta=cfg.hparams.beta
    )

    if num_devices > 1:
        # Create a Sharding object to distribute a value across devices:
        mesh = jax.make_mesh(
            axis_shapes=(num_devices,),
            axis_names=('batch',)
        )

    for indices in tqdm(
        iterable=data_loader,
        total=len(log_p_y) // cfg.hparams.batch_size_em,
        desc=' re-label',
        leave=False,
        ncols=80,
        colour='green',
        position=1,
        disable=not cfg.data_loading.progress_bar
    ):
        key, _ = jax.random.split(key=key, num=2)

        # extract the corresponding noisy labels
        log_p_y_ = log_p_y[indices]  # (B, C)
        log_mult_prob_ = log_mult_prob[indices]  # (B, C, C)

        # extract coding coefficients
        log_nn_coding = jnp.log(coding_matrix[indices])  # (B, K)

        # region APPROXIMATE p(y | x) through nearest-neighbours
        nn_idx_ = nn_idx[indices]
        log_p_y_nn = log_p_y[nn_idx_]  # (B, K, C)
        log_mult_prob_nn = log_mult_prob[nn_idx_]  # (B, K, C, C)
        log_mult_prob_nn = jnp.reshape(
            a=log_mult_prob_nn,
            shape=(log_mult_prob_nn.shape[0], -1, log_mult_prob_nn.shape[-1])
        )  # (B, K*C, C)

        # calculate the mixture coefficients of other mixtures
        # induced by nearest-neighbours
        log_mixture_coefficient_nn = jnp.log(1 - cfg.hparams.mu) \
            + log_nn_coding[:, :, None] + log_p_y_nn  # (B, K, C)
        log_mixture_coefficient_nn = jnp.reshape(
            a=log_mixture_coefficient_nn,
            shape=(log_mixture_coefficient_nn.shape[0], -1)
        )  # (B, K*C)

        log_mixture_coefficient_x = jnp.log(cfg.hparams.mu) \
            + log_p_y_  # (B, C)

        log_mixture_coefficient = jnp.concatenate(
            arrays=(log_mixture_coefficient_x, log_mixture_coefficient_nn),
            axis=-1
        )  # (B, (K + 1) * C)
        log_multinomial_probs = jnp.concatenate(
            arrays=(log_mult_prob_, log_mult_prob_nn),
            axis=1
        )  # (B, (K + 1) * C, C)
        # endregion

        # distributed computation
        if (num_devices > 1) and (len(indices) % num_devices == 0):
            log_mixture_coefficient = jax.device_put(
                x=log_mixture_coefficient,
                device=NamedSharding(mesh=mesh, spec=PartitionSpec('batch'))
            )
            log_multinomial_probs = jax.device_put(
                x=log_multinomial_probs,
                device=NamedSharding(mesh=mesh, spec=PartitionSpec('batch'))
            )

        # predict the clean label distribution using EM
        log_p_y_temp, log_mult_prob_temp = get_p_y_fn(
            log_mixture_coefficient,
            log_multinomial_probs,
            key
        )

        # if jnp.any(a=jnp.isnan(log_p_y_temp)):
        #     raise ValueError('NaN is detected after running EM')

        # update the noisy labels
        log_p_y_new = log_p_y_new.at[indices].set(values=log_p_y_temp)
        log_mult_prob_new = (
            log_mult_prob_new.at[indices]
            .set(values=log_mult_prob_temp)
        )

    return dict(p_y=jnp.exp(log_p_y_new), log_mult_prob=log_mult_prob_new)


def relabel_data_sparse(
        mult_mixture: dict[str, Any],
        nn_idx: jax.Array,
        coding_matrix: jax.Array,
        cfg: DictConfig) -> dict[str, Any]:
    """Perform EM to infer the p(y | x) and p(yhat | x, y)
    In this implementation, both p(y | x) and p(yhat | x, y) are sparse tensors
    and assumed that their dense dimensions are the same across all samples.
    This assumption enables the vectorization in the implementation.

    Args:
        log_p_y: a row-wise matrix containing the "cleaner"
            label distributions of samples  (N , C)
        log_mult_prob: a tensor containing the transition matrices
            of samples  (N, C, C)
        nn_idx: a matrix where each row contains the indices of
            nearest-neighbours corresponding to row (sample) id  (N, K)
        coding_matrix: a matrix where each row contains
            the coding coefficient (normalised similarity)  (N, K)
        args: contains configuration parameters

    Returns:
        p_y: a sparse matrix where each row is p(y | x)  # (N, C)
        mult_prob: a sparse 3-d tensor where each matrix is
            p(yhat | x, y)  # (N, C , C)
    """
    num_devices = jax.device_count(backend='gpu')

    # initialize new p(y | x) and p(yhat | x, y)
    p_y_new = dict(
        data=jnp.empty(
            shape=(len(nn_idx), cfg.hparams.num_approx_classes),
            dtype=jnp.float32
        ),
        indices=jnp.empty(
            shape=(len(nn_idx), cfg.hparams.num_approx_classes),
            dtype=jnp.int32
        )
    )
    log_mult_prob_new = jnp.empty(
        shape=(
            len(nn_idx),
            cfg.hparams.num_approx_classes,
            cfg.dataset.num_classes
        ),
        dtype=jnp.float32
    )

    seed = random.randint(a=0, b=1_000)
    data_loader = (
        grain.MapDataset.range(start=0, stop=len(nn_idx), step=1)
        .shuffle(seed=seed)  # Shuffles globally.
        .map(lambda x: x)  # Maps each element.
        .batch(
            batch_size=cfg.hparams.batch_size_em
        )  # Batches consecutive elements.
    )

    key = jax.random.key(seed=seed)

    get_p_y_fn = partial(
        get_p_y,
        num_noisy_labels_per_sample=cfg.hparams.num_noisy_labels_per_sample,
        num_multinomial_samples=cfg.hparams.num_multinomial_samples,
        num_classes=cfg.dataset.num_classes,
        num_em_iter=cfg.hparams.num_em_iter,
        alpha=cfg.hparams.alpha,
        beta=cfg.hparams.beta
    )

    # region SHARDING
    if num_devices > 1:
        # Create a Sharding object to distribute a value across devices:
        mesh = jax.make_mesh(
            axis_shapes=(num_devices,),
            axis_names=('batch',)
        )
    
    # endergion

    for indices in tqdm(
        iterable=data_loader,
        desc=' re-label',
        leave=False,
        ncols=80,
        colour='green',
        position=1,
        disable=not cfg.data_loading.progress_bar
    ):
        key, _ = jax.random.split(key=key, num=2)

        # extract the corresponding noisy labels
        p_y_batch = mult_mixture['p_y']['data'][indices]  # (B, C0)

        log_mult_prob_ = mult_mixture['log_mult_prob'][indices]  # (B, C0, C)

        # extract coding coefficients
        nn_coding = coding_matrix[indices]  # (B, K)

        # region APPROXIMATE p(y | x) through nearest-neighbours
        nn_idx_ = nn_idx[indices]  # (B, K)

        p_y_nn = mult_mixture['p_y']['data'][nn_idx_]  # (B, K, C0)

        # get the mult components of shape (B, K, C0, C)
        log_mult_prob_nn = mult_mixture['log_mult_prob'][nn_idx_]
        log_mult_prob_nn = jnp.reshape(
            a=log_mult_prob_nn,
            shape=(
                log_mult_prob_nn.shape[0],
                -1,
                log_mult_prob_nn.shape[-1]
            )
        )  # (B, K*C0, C)

        # coefficient of neighbor samples (B, K, C0)
        mixture_coefficient_nn = (1 - cfg.hparams.mu) \
            * nn_coding[:, :, None] \
            * p_y_nn
        mixture_coefficient_nn = jnp.reshape(
            a=mixture_coefficient_nn,
            shape=(len(indices), -1)
        )  # (B, K*C0)

        # mixture coefficient of samples of interest
        mixture_coefficient_x = cfg.hparams.mu * p_y_batch  # (B, C0)

        mixture_coefficient = jnp.concatenate(
            arrays=(mixture_coefficient_x, mixture_coefficient_nn),
            axis=-1
        )  # (B, (K + 1) * C0)

        log_multinomial_probs = jnp.concatenate(
            arrays=(log_mult_prob_, log_mult_prob_nn),
            axis=1
        )  # (B, (K + 1) * C0, C)
        # endregion

        # distributed computation
        if (num_devices > 1) and (len(indices) % num_devices == 0):
            mixture_coefficient = jax.device_put(
                x=mixture_coefficient,
                device=NamedSharding(mesh=mesh, spec=PartitionSpec('batch'))
            )
            log_multinomial_probs = jax.device_put(
                x=log_multinomial_probs,
                device=NamedSharding(mesh=mesh, spec=PartitionSpec('batch'))
            )

        # predict the clean label distribution using EM
        log_p_y_temp, log_mult_prob_temp = get_p_y_fn(
            jnp.log(mixture_coefficient),
            log_multinomial_probs,
            key
        )

        p_y_temp = jnp.exp(log_p_y_temp)

        # if jnp.any(a=jnp.isnan(log_p_y_temp)):
        #     raise ValueError('NaN is detected after running EM')

        # region TRUNCATE the result
        top_k_data, top_k_indices = jax.lax.top_k(
            operand=p_y_temp,
            k=cfg.hparams.num_approx_classes
        )  # (B, C0)

        # truncate the mixture vector and normalise top_k_data
        top_k_data /= jnp.sum(a=top_k_data, axis=-1, keepdims=True)  # (B, C0)

        # append data and corresponding indices
        p_y_new['data'] = (
            p_y_new['data']
            .at[indices]
            .set(values=top_k_data)
        )

        p_y_new['indices'] = (
            p_y_new['indices']
            .at[indices]
            .set(values=top_k_indices)
        )

        # truncate the multinominal components
        log_mult_prob_truncated = log_mult_prob_temp[
            jnp.arange(indices.size)[:, None],
            top_k_indices
        ]
        # endregion

        # update the noisy labels
        log_mult_prob_new = log_mult_prob_new.at[indices].set(
            values=log_mult_prob_truncated
        )

    return dict(p_y=p_y_new, log_mult_prob=log_mult_prob_new)


def get_nn_coding_matrix(
    model: nnx.Module,
    sample_indices: jax.Array,
    cfg: DictConfig
) -> tuple[jax.Array, jax.Array]:
    """
    """
    num_devices = jax.device_count()

    sub_datasource = ImageDataSource(
        annotation_file=cfg.dataset.train_file,
        root=cfg.dataset.root,
        idx_list=sample_indices
    )

    sub_dataloader = initialize_dataloader(
        data_source=sub_datasource,
        num_epochs=1,
        seed=0,
        shuffle=False,
        batch_size=cfg.training.batch_size,
        resize=cfg.data_augmentation.resize,
        padding_px=cfg.data_augmentation.padding_px,
        crop_size=cfg.data_augmentation.crop_size,
        mean=cfg.data_augmentation.mean,
        std=cfg.data_augmentation.std,
        p_flip=cfg.data_augmentation.prob_random_flip,
        num_workers=cfg.data_loading.num_workers,
        num_threads=cfg.data_loading.num_threads,
        prefetch_size=cfg.data_loading.prefetch_size
    )

    # extract features
    features = extract_features(
        model=model,
        dataloader=sub_dataloader,
        cfg=cfg
    )

    # convert to f32
    features = features.astype(dtype=jnp.float32)

    if num_devices > 1:
        # Create a Sharding object to distribute a value across devices:
        mesh = jax.make_mesh(
            axis_shapes=(num_devices,),
            axis_names=('batch',)
        )

        features = jax.device_put(
            x=features,
            device=NamedSharding(mesh=mesh, spec=PartitionSpec('batch'))
        )

    # find K nearest-neighbours
    knn_indices = get_knn_indices(
        xb=features,
        num_nn=cfg.hparams.num_nearest_neighbors,
        ids=sample_indices
    )

    # calculate coding mtrices
    coding_matrix = get_batch_local_affine_coding(
        samples=features,
        knn_indices=knn_indices
    )

    return knn_indices, coding_matrix


@hydra.main(version_base=None, config_path='conf', config_name='conf')
def main(cfg: DictConfig) -> None:
    """Main function
    """
    logging.info('Identifiability in noisy label learning')
    logging.info(f'Dataset = {cfg.dataset.name}')
    logging.info(f'Num classes = {cfg.dataset.num_classes}')
    logging.info(f'Num approx classes = {cfg.hparams.num_approx_classes}')

    # region ENVIRONMENT
    jax.config.update('jax_disable_jit', not cfg.jax.jit)
    jax.config.update('jax_platforms', cfg.jax.platform)
    jax.config.update('jax_default_device', jax.devices()[-1])

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)
    # endregion

    OmegaConf.update(
        cfg=cfg,
        key='hparams.num_noisy_labels_per_sample',
        value=eval(cfg.hparams.num_noisy_labels_per_sample)
    )

    match cfg.hparams.num_approx_classes:
        case None:  # no approximation
            get_noisy_labels_fn = get_soft_noisy_labels
            relabel_data_fn = relabel_data
        case _:
            get_noisy_labels_fn = get_sparse_noisy_labels
            relabel_data_fn = relabel_data_sparse

    # region DATASETS
    source_train, source_test = get_data_sources(
        train_file=cfg.dataset.train_file,
        test_file=cfg.dataset.test_file,
        root=cfg.dataset.root
    )

    OmegaConf.set_struct(conf=cfg, value=True)
    OmegaConf.update(
        cfg=cfg,
        key='dataset.length.train',
        value=len(source_train),
        force_add=True
    )
    OmegaConf.update(
        cfg=cfg,
        key='dataset.length.test',
        value=len(source_test),
        force_add=True
    )
    # endregion

    # region MODELS
    model_1 = FeatureEmbeddedModel(
        feature_extractor=hydra.utils.instantiate(config=cfg.model)(
            num_classes=cfg.hparams.feature_dim,
            rngs=nnx.Rngs(jax.random.PRNGKey(seed=random.randint(a=0, b=100))),
            dropout_rate=cfg.training.dropout_rate,
            dtype=eval(cfg.jax.dtype)
        ),
        num_classes=cfg.dataset.num_classes,
        dtype=eval(cfg.jax.dtype),
        rngs=nnx.Rngs(jax.random.PRNGKey(seed=random.randint(a=0, b=100)))
    )
    optimizer_1 = nnx.Optimizer(
        model=model_1,
        tx=init_tx(
            dataset_length=len(source_train),
            lr=cfg.training.lr,
            batch_size=cfg.training.batch_size,
            num_epochs=cfg.training.num_epochs + cfg.training.num_epochs_warmup,
            weight_decay=cfg.training.weight_decay,
            momentum=cfg.training.momentum,
            clipped_norm=cfg.training.clipped_norm,
            key=random.randint(a=0, b=100)
        ),
        wrt=nnx.Param
    )

    model_2 = FeatureEmbeddedModel(
        feature_extractor=hydra.utils.instantiate(config=cfg.model)(
            num_classes=cfg.hparams.feature_dim,
            rngs=nnx.Rngs(jax.random.PRNGKey(seed=random.randint(a=0, b=100))),
            dropout_rate=cfg.training.dropout_rate,
            dtype=eval(cfg.jax.dtype)
        ),
        num_classes=cfg.dataset.num_classes,
        dtype=eval(cfg.jax.dtype),
        rngs=nnx.Rngs(jax.random.PRNGKey(seed=random.randint(a=0, b=100)))
    )
    optimizer_2 = nnx.Optimizer(
        model=model_2,
        tx=init_tx(
            dataset_length=len(source_train),
            lr=cfg.training.lr,
            batch_size=cfg.training.batch_size,
            num_epochs=cfg.training.num_epochs + cfg.training.num_epochs_warmup,
            weight_decay=cfg.training.weight_decay,
            momentum=cfg.training.momentum,
            clipped_norm=cfg.training.clipped_norm,
            key=random.randint(a=0, b=100)
        ),
        wrt=nnx.Param
    )
    # endregion

    # region LOAD NOISY LABEL DISTRIBUTION
    mult_mixture_1 = get_noisy_labels_fn(
        data_source=source_train,
        num_classes=cfg.dataset.num_classes,
        progress_bar_flag=cfg.data_loading.progress_bar
    )
    mult_mixture_2 = {key: mult_mixture_1[key] for key in mult_mixture_1}
    # endregion

    # region EXPERIMENT TRACKING
    # create log folder if it does not exist
    logging.info(msg='Initialise directory to store logs')
    if not os.path.exists(path=cfg.experiment.logdir):
        logging.info(
            msg=f'Logging folder "{cfg.experiment.logdir}" not found.'
        )
        Path(cfg.experiment.logdir).mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(uri=cfg.experiment.tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.experiment.name)
    mlflow.disable_system_metrics_logging()
    # endregion

    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=50,
        max_to_keep=10,
        step_format_fixed_length=3,
        enable_async_checkpointing=False
    )

    with mlflow.start_run(
            run_id=cfg.experiment.run_id,
            log_system_metrics=False) as mlflow_run, \
        ocp.CheckpointManager(
            directory=os.path.join(
                os.getcwd(),
                cfg.experiment.logdir,
                cfg.experiment.name,
                mlflow_run.info.run_id
            ),
            item_names=('data_1', 'data_2'),
            options=ckpt_options) as ckpt_mngr:

        if cfg.experiment.run_id is None:
            start_epoch_id = 0

            # log hyper-parameters
            mlflow.log_params(
                params=flatten_dict(
                    xs=OmegaConf.to_container(cfg=cfg),
                    sep='.'
                )
            )

            # log source code
            mlflow.log_artifact(
                local_path=os.path.abspath(path=__file__),
                artifact_path='source_code'
            )
        else:
            start_epoch_id = ckpt_mngr.latest_step()

            checkpoint = ckpt_mngr.restore(
                step=start_epoch_id,
                args=ocp.args.Composite(
                    data_1=ocp.args.StandardRestore(
                        item={
                            'state': nnx.state(model_1),
                            'mult_mixture': mult_mixture_1
                        }
                    ),
                    data_2=ocp.args.StandardRestore(
                        item={
                            'state': nnx.state(model_2),
                            'mult_mixture': mult_mixture_2
                        }
                    )
                )
            )

            nnx.update(model_1, checkpoint['data_1']['state'])
            mult_mixture_1 = checkpoint['data_1']['mult_mixture']

            nnx.update(model_2, checkpoint['data_2']['state'])
            mult_mixture_2 = checkpoint['data_2']['mult_mixture']

            del checkpoint

        # region DATA LOADERS
        dataloader_train_fn = partial(
            initialize_dataloader,
            shuffle=True,
            batch_size=cfg.training.batch_size,
            resize=cfg.data_augmentation.resize,
            padding_px=cfg.data_augmentation.padding_px,
            crop_size=cfg.data_augmentation.crop_size,
            mean=cfg.data_augmentation.mean,
            std=cfg.data_augmentation.std,
            p_flip=cfg.data_augmentation.prob_random_flip,
            num_workers=cfg.data_loading.num_workers,
            num_threads=cfg.data_loading.num_threads,
            prefetch_size=cfg.data_loading.prefetch_size
        )
        iter_dataloader_train_1 = dataloader_train_fn(
            data_source=source_train,
            num_epochs=cfg.training.num_epochs - start_epoch_id + cfg.training.num_epochs_warmup + 1,
            seed=random.randint(a=0, b=1_000)
        )
        iter_dataloader_train_1 = iter(iter_dataloader_train_1)

        iter_dataloader_train_2 = dataloader_train_fn(
            data_source=source_train,
            num_epochs=cfg.training.num_epochs - start_epoch_id + cfg.training.num_epochs_warmup + 1,
            seed=random.randint(a=0, b=1_000)
        )
        iter_dataloader_train_2 = iter(iter_dataloader_train_2)

        data_loader_test = initialize_dataloader(
            data_source=source_test,
            num_epochs=1,
            shuffle=False,
            seed=0,
            batch_size=cfg.training.batch_size,
            resize=cfg.data_augmentation.crop_size,
            padding_px=None,
            crop_size=None,
            mean=cfg.data_augmentation.mean,
            std=cfg.data_augmentation.std,
            p_flip=None,
            is_color_img=True,
            num_workers=cfg.data_loading.num_workers,
            num_threads=cfg.data_loading.num_threads,
            prefetch_size=cfg.data_loading.prefetch_size
        )
        # endregion

        # region WARMUP
        if start_epoch_id == 0:
            for epoch_id in tqdm(
                iterable=range(cfg.training.num_epochs_warmup),
                desc='warmup',
                ncols=80,
                leave=True,
                position=0,
                colour='green',
                disable=not cfg.data_loading.progress_bar
            ):
                model_1, optimizer_1, loss_1 = train_cross_entropy(
                    model=model_1,
                    optimizer=optimizer_1,
                    dataloader=iter_dataloader_train_1,
                    p_y=mult_mixture_2['p_y'],
                    cfg=cfg
                )

                model_2, optimizer_2, loss_2 = train_cross_entropy(
                    model=model_2,
                    optimizer=optimizer_2,
                    dataloader=iter_dataloader_train_2,
                    p_y=mult_mixture_1['p_y'],
                    cfg=cfg
                )

                acc_1 = evaluate(
                    model=model_1,
                    dataloader=data_loader_test,
                    cfg=cfg
                )
                acc_2 = evaluate(
                    model=model_2,
                    dataloader=data_loader_test,
                    cfg=cfg
                )

                mlflow.log_metrics(
                    metrics={
                        'warmup/loss_1': loss_1,
                        'warmup/loss_2': loss_2,
                        'warmup/accuracy_1': acc_1,
                        'warmup/accuracy_2': acc_2
                    },
                    step=epoch_id + 1,
                    synchronous=False
                )
        # endergion

        # region TRAINING
        for epoch_id in tqdm(
            iterable=range(start_epoch_id, cfg.training.num_epochs, 1),
            desc='progress',
            ncols=80,
            leave=True,
            position=0,
            colour='green',
            disable=not cfg.data_loading.progress_bar
        ):
            if (epoch_id + 1) % cfg.hparams.relabeling_every_n_epochs == 0 \
                or epoch_id == 0:

                idx_loader1 = (
                    grain.MapDataset.range(
                        start=0,
                        stop=cfg.dataset.length.train,
                        step=1
                    )
                    .shuffle(
                        seed=random.randint(a=0, b=100_000)
                    )  # Shuffles globally.
                    .map(lambda x: x)  # Maps each element.
                    .batch(
                        batch_size=cfg.hparams.num_samples_search_knn
                    )  # Batches consecutive elements.
                )
                idx_loader2 = (
                    grain.MapDataset.range(
                        start=0,
                        stop=cfg.dataset.length.train,
                        step=1
                    )
                    .shuffle(
                        seed=random.randint(a=0, b=100_000)
                    )  # Shuffles globally.
                    .map(lambda x: x)  # Maps each element.
                    .batch(
                        batch_size=cfg.hparams.num_samples_search_knn
                    )  # Batches consecutive elements.
                )

                # initialize nearest neighbor matrix and coding matrix
                nn_matrix_1 = jnp.zeros(
                    shape=(
                        cfg.dataset.length.train,
                        cfg.hparams.num_nearest_neighbors
                    ),
                    dtype=jnp.int32
                )  # (N, K)
                coding_matrix_1 = jnp.zeros_like(
                    a=nn_matrix_1,
                    dtype=jnp.float32
                )  # (N, K)

                nn_matrix_2 = jnp.zeros_like(a=nn_matrix_1, dtype=jnp.int32)
                coding_matrix_2 = jnp.zeros_like(
                    a=nn_matrix_1,
                    dtype=jnp.float32
                )

                for sample_indices_1, sample_indices_2 in tqdm(
                    iterable=zip(idx_loader1, idx_loader2),
                    total=int(jnp.ceil(cfg.dataset.length.train / cfg.hparams.num_samples_search_knn)),
                    desc='Neighboring',
                    ncols=80,
                    leave=False,
                    position=1,
                    colour='green',
                    disable=not cfg.data_loading.progress_bar
                ):
                    # region FIND K-NN and CODING MATRIX
                    nn_matrix_1_temp, coding_matrix_1_temp = \
                        get_nn_coding_matrix(
                            model=model_1,
                            sample_indices=sample_indices_1,
                            cfg=cfg
                        )

                    nn_matrix_1 = (
                        nn_matrix_1.at[sample_indices_1]
                        .set(values=nn_matrix_1_temp.astype(jnp.int32))
                    )
                    coding_matrix_1 = (
                        coding_matrix_1.at[sample_indices_1]
                        .set(values=coding_matrix_1_temp)
                    )

                    nn_matrix_2_temp, coding_matrix_2_temp = \
                        get_nn_coding_matrix(
                            model=model_2,
                            sample_indices=sample_indices_2,
                            cfg=cfg
                        )

                    nn_matrix_2 = (
                        nn_matrix_2.at[sample_indices_2]
                        .set(values=nn_matrix_2_temp.astype(jnp.int32))
                    )
                    coding_matrix_2 = (
                        coding_matrix_2.at[sample_indices_2]
                        .set(values=coding_matrix_2_temp)
                    )
                    # endregion

                # region RE-LABEL
                mult_mixture_1 = relabel_data_fn(
                    mult_mixture=mult_mixture_1,
                    nn_idx=nn_matrix_1,
                    coding_matrix=coding_matrix_1,
                    cfg=cfg
                )

                mult_mixture_2 = relabel_data_fn(
                    mult_mixture=mult_mixture_2,
                    nn_idx=nn_matrix_2,
                    coding_matrix=coding_matrix_2,
                    cfg=cfg
                )
                # endregion

            # region TRAIN models on relabelled data
            model_1, optimizer_1, loss_1 = train_cross_entropy(
                model=model_1,
                optimizer=optimizer_1,
                dataloader=iter_dataloader_train_1,
                p_y=mult_mixture_2['p_y'],
                cfg=cfg
            )

            model_2, optimizer_2, loss_2 = train_cross_entropy(
                model=model_2,
                optimizer=optimizer_2,
                dataloader=iter_dataloader_train_2,
                p_y=mult_mixture_1['p_y'],
                cfg=cfg
            )
            # endregion

            # save checkpoint
            ckpt_mngr.save(
                step=epoch_id + 1,
                args=ocp.args.Composite(
                    data_1=ocp.args.StandardSave({
                        'state': nnx.state(model_1),
                        'mult_mixture': mult_mixture_1
                    }),
                    data_2=ocp.args.StandardSave({
                        'state': nnx.state(model_2),
                        'mult_mixture': mult_mixture_2
                    })
                )
            )

            # evaluate
            acc_1 = evaluate(
                model=model_1,
                dataloader=data_loader_test,
                cfg=cfg
            )
            acc_2 = evaluate(
                model=model_2,
                dataloader=data_loader_test,
                cfg=cfg
            )

            mlflow.log_metrics(
                metrics={
                    'loss/model_1': loss_1,
                    'loss/model_2': loss_2,
                    'accuracy/model_1': acc_1,
                    'accuracy/model_2': acc_2
                },
                step=epoch_id + 1,
                synchronous=False
            )
        # endergion

    logging.info(msg='Training is completed.')


if __name__ == '__main__':
    logger_current = logging.getLogger(name=__name__)
    logger_current.setLevel(level=logging.INFO)
    main()
