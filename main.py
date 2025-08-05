import jax
import jax.numpy as jnp
from jax.experimental import sparse

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
    progress_bar_flag: bool = True,
    **kwargs
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
    log_p_y = jnp.log(p_y)

    log_mult_prob = jnp.eye(N=num_classes)  # (C, C)
    log_mult_prob = jnp.tile(
        A=log_mult_prob[None, :, :],
        reps=(len(data_source), 1, 1)
    )  # (N, C, C)

    # add random noise
    log_mult_prob = 10 * log_mult_prob \
        + jax.random.uniform(key=key, shape=log_mult_prob.shape)
    log_mult_prob = jax.nn.log_softmax(x=log_mult_prob, axis=-1)

    return dict(log_p_y=log_p_y, log_mult_prob=log_mult_prob)


def get_sparse_noisy_labels(
    data_source: grain.RandomAccessDataSource,
    num_classes: int,
    num_approx_classes: int,
    seed: int | None = None,
    progress_bar_flag: bool = False
) -> dict[str, jax.Array | sparse.BCOO]:
    """load the noisy labels as multinomial mixtures in the sparse setting

    Args:
        data_source:
        num_classes: the total number of classes
        num_approx_classes: the actual number of classes (non-zero elements)

    Return:
        mult_mixture: a dictionary contains:
            - log_p_y: logarithm of mixture coefficient or clean label dist.
            - log_mult_prob:
    """
    # initialise parameters of multinomial mixture model
    log_mult_prob = jnp.empty(
        shape=(len(data_source), num_approx_classes, num_classes),
        dtype=jnp.float32
    )

    log_p_y_indices = []
    log_p_y_data = []

    for i in tqdm(
        iterable=range(len(data_source)),
        desc='load noisy labels',
        leave=False,
        ncols=79,
        position=0,
        colour='green',
        disable=not progress_bar_flag
    ):
        key = jax.random.key(
            seed=seed if seed is not None else random.randint(a=0, b=1_000)
        )
        random_noise = 0.1 * jax.random.uniform(
            key=key,
            shape=(num_classes,)
        )

        # convert integer numbers to one-hot vectors
        noisy_label = jax.nn.one_hot(
            x=data_source[i]['label'],
            num_classes=num_classes,
            dtype=jnp.float32
        )  # (B, C)

        # randomly pick some classes (including the noisy label class)
        _, class_indices = jax.lax.top_k(
            operand=noisy_label + random_noise,
            k=num_approx_classes
        )  # (B, C0)

        # reduce from num_classes to num_approx_classes
        few_class_noisy_label = jnp.take_along_axis(
            arr=noisy_label,
            indices=class_indices,
            axis=-1
        )

        # normalize to sum to 1
        few_class_noisy_label /= jnp.sum(a=few_class_noisy_label, axis=0)

        few_class_noisy_label = optax.smooth_labels(
            labels=few_class_noisy_label,
            alpha=0.1
        )

        log_p_y_data_temp = jnp.log(few_class_noisy_label)
        log_p_y_indices.append(class_indices)
        log_p_y_data.append(log_p_y_data_temp)

        # log of multinomial probability vector
        log_mult_prob_temp = jnp.log(optax.smooth_labels(
            labels=jax.nn.one_hot(
                x=class_indices,
                num_classes=num_classes,
                dtype=jnp.float32
            ),
            alpha=0.1
        ))
        log_mult_prob = log_mult_prob.at[i].set(values=log_mult_prob_temp)

    log_p_y_data = jnp.stack(arrays=log_p_y_data, axis=0)  # (N, C0)
    log_p_y_indices = jnp.stack(
        arrays=log_p_y_indices,
        axis=0
    )  # (N, C0)
    log_p_y_indices = jnp.expand_dims(a=log_p_y_indices, axis=-1)

    # sparse clean label distribution
    log_p_y = sparse.BCOO(
        args=(log_p_y_data, log_p_y_indices),
        shape=(len(data_source), num_classes)
    )  # (N, C)

    return dict(log_p_y=log_p_y, log_mult_prob=log_mult_prob)


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
    p_y: jax.Array,
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


@partial(
    jax.jit,
    static_argnames=(
        'num_noisy_labels_per_sample',
        'num_multinomial_samples',
        'num_classes',
        'batch_size_em',
        'num_em_iter',
        'alpha',
        'beta'
    )
)
def _get_p_y(
    log_mixture_coefficients: jax.Array,
    log_multinomial_probs: jax.Array,
    key: jax._src.prng.PRNGKeyArray,
    num_noisy_labels_per_sample: int,
    num_multinomial_samples: int,
    num_classes: int,
    batch_size_em: int,
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
    log_mixture_coefficients: jax.Array,
    log_multinomial_probs: jax.Array,
    cfg: DictConfig
) -> tuple[jax.Array, jax.Array]:
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
        key=jax.random.key(seed=random.randint(a=0, b=100_000)),
        num_noisy_labels_per_sample=cfg.hparams.num_noisy_labels_per_sample,
        num_multinomial_samples=cfg.hparams.num_multinomial_samples,
        num_classes=cfg.dataset.num_classes,
        batch_size_em=cfg.hparams.batch_size_em,
        num_em_iter=cfg.hparams.num_em_iter,
        alpha=cfg.hparams.alpha,
        beta=cfg.hparams.beta
    )


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
    log_p_y = mult_mixture['log_p_y']
    log_mult_prob = mult_mixture['log_mult_prob']

    # initialize new p(y | x) and p(yhat | x, y)
    log_p_y_new = log_p_y + 0.
    log_mult_prob_new = log_mult_prob + 0.

    data_loader = (
        grain.MapDataset.range(start=0, stop=len(log_p_y), step=1)
        .shuffle(seed=random.randint(a=0, b=100_000))  # Shuffles globally.
        .map(lambda x: x)  # Maps each element.
        .batch(
            batch_size=cfg.hparams.batch_size_em
        )  # Batches consecutive elements.
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

        # predict the clean label distribution using EM
        log_p_y_temp, log_mult_prob_temp = get_p_y(
            log_mixture_coefficients=log_mixture_coefficient,
            log_multinomial_probs=log_multinomial_probs,
            cfg=cfg
        )

        # if jnp.any(a=jnp.isnan(log_p_y_temp)):
        #     raise ValueError('NaN is detected after running EM')

        # update the noisy labels
        log_p_y_new = log_p_y_new.at[indices].set(values=log_p_y_temp)
        log_mult_prob_new = (
            log_mult_prob_new.at[indices]
            .set(values=log_mult_prob_temp)
        )

    return dict(log_p_y=log_p_y_new, log_mult_prob=log_mult_prob_new)


def relabel_data_sparse(
        mult_mixture: dict[str, sparse.BCOO | jax.Array],
        nn_idx: jax.Array,
        coding_matrix: jax.Array,
        cfg: DictConfig) -> dict[str, sparse.BCOO | jax.Array]:
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

    Returns:
        p_y: a sparse matrix where each row is p(y | x)  # (N, C)
        mult_prob: a sparse 3-d tensor where each matrix is p(yhat | x, y)  # (N, C , C)
    """
    # initialize new p(y | x) and p(yhat | x, y)
    log_p_y_new = dict(data=[], indices=[])
    log_mult_prob_new = jnp.empty_like(
        prototype=mult_mixture['log_mult_prob'],
        dtype=jnp.float32
    )

    data_loader = (
        grain.MapDataset.range(start=0, stop=len(nn_idx), step=1)
        .shuffle(seed=random.randint(a=0, b=100_000))  # Shuffles globally.
        .map(lambda x: x)  # Maps each element.
        .batch(
            batch_size=cfg.hparams.batch_size_em
        )  # Batches consecutive elements.
    )

    for indices in tqdm(
        iterable=data_loader,
        total=len(nn_idx) // cfg.hparams.batch_size_em,
        desc=' re-label',
        leave=False,
        ncols=80,
        colour='green',
        position=1,
        disable=not cfg.data_loading.progress_bar
    ):
        # extract the corresponding noisy labels
        log_p_y_ = sparse.BCOO(
            args=(
                mult_mixture['log_p_y'].data[indices],
                mult_mixture['log_p_y'].indices[indices]
            ),
            shape=(indices.size, cfg.dataset.num_classes)
        )  # (B, C)

        log_mult_prob_ = mult_mixture['log_mult_prob'][indices]  # (B, C0, C)

        # extract coding coefficients
        log_nn_coding = jnp.log(coding_matrix[indices])  # (B, K)

        # region APPROXIMATE p(y | x) through nearest-neighbours
        nn_idx_ = nn_idx[indices]  # (B, K)

        log_p_y_nn = sparse.BCOO(
            args=(
                mult_mixture['log_p_y'].data[nn_idx_],
                mult_mixture['log_p_y'].indices[nn_idx_]
            ),
            shape=(
                indices.size,
                cfg.hparams.num_nearest_neighbors,
                cfg.dataset.num_classes
            )
        )  # (B, K, C)

        log_mult_prob_nn = mult_mixture['log_mult_prob'][nn_idx_]  # (B, K, C0, C)
        log_mult_prob_nn = jnp.reshape(
            a=log_mult_prob_nn,
            shape=(
                log_mult_prob_nn.shape[0],
                -1,
                log_mult_prob_nn.shape[-1]
            )
        )  # (B, K*C0, C)

        # calculate the mixture coefficients of other mixtures
        # induced by nearest-neighbours
        log_nn_coding_broadcast = jnp.broadcast_to(
            array=log_nn_coding[:, :, None],
            shape=log_p_y_nn.shape
        )  # (B, K, C)
        log_nn_coding_broadcast = log_nn_coding_broadcast \
            + jnp.log(1 - cfg.hparams.mu)  # (B, K, C)
        log_nn_coding_sparse = sparse.bcoo_extract(
            sparr=log_p_y_nn, arr=log_nn_coding_broadcast
        )

        # mixture coefficients of NN samples
        log_mixture_coefficient_nn = sparse.BCOO(
            args=(
                log_nn_coding_sparse.data + log_p_y_nn.data,
                log_p_y_nn.indices
            ),
            shape=log_p_y_nn.shape
        )  # (B, K, C)

        # update n_batch to 1 in order to reshape
        log_mixture_coefficient_nn = sparse.bcoo_update_layout(
            mat=log_mixture_coefficient_nn,
            n_batch=1
        )

        # flatten the last dimension/axis
        log_mixture_coefficient_nn = sparse.bcoo_reshape(
            mat=log_mixture_coefficient_nn,
            new_sizes=(
                log_mixture_coefficient_nn.shape[0],
                log_mixture_coefficient_nn[0].size
            )
        )  # (B, K*C)

        # mixture coefficient of samples of interest
        log_mixture_coefficient_x = sparse.BCOO(
            args=(jnp.log(cfg.hparams.mu) + log_p_y_.data, log_p_y_.indices),
            shape=log_p_y_.shape
        )  # (B, C)

        # mixture coefficient used in the approximation of label distribution
        log_mixture_coefficient_sparse = sparse.bcoo_concatenate(
            operands=(log_mixture_coefficient_x, log_mixture_coefficient_nn),
            dimension=1
        )  # (B, (K + 1) * C)

        log_mixture_coefficient = log_mixture_coefficient_sparse.data  # (B, (K + 1) * C0)
        # log_mixture_coefficient = jnp.reshape(
        #     a=log_mixture_coefficient,
        #     shape=(indices.size, -1)
        # )  # (B, (K + 1) * C0)

        log_multinomial_probs = jnp.concatenate(
            arrays=(log_mult_prob_, log_mult_prob_nn),
            axis=1
        )  # (B, (K + 1) * C0, C)
        # endregion

        # predict the clean label distribution using EM
        log_p_y_temp, log_mult_prob_temp = get_p_y(
            log_mixture_coefficients=log_mixture_coefficient,
            log_multinomial_probs=log_multinomial_probs,
            cfg=cfg
        )

        # if jnp.any(a=jnp.isnan(log_p_y_temp)):
        #     raise ValueError('NaN is detected after running EM')

        # region TRUNCATE the result
        top_k_data, top_k_indices = jax.lax.top_k(
            operand=log_p_y_temp,
            k=cfg.hparams.num_approx_classes
        )  # (B, C0)

        # truncate the mixture vector
        # normalise top_k_data
        top_k_data = jax.nn.log_softmax(x=top_k_data, axis=-1)  # (B, C0)

        log_p_y_new['data'].append(top_k_data),
        log_p_y_new['indices'].append(
            jnp.expand_dims(a=top_k_indices, axis=-1)
        )

        # truncate the multinominal components
        log_mult_prob_truncated = log_mult_prob_temp[jnp.arange(indices.size)[:, None], top_k_indices]
        # endregion

        # update the noisy labels
        log_mult_prob_new = log_mult_prob_new.at[indices].set(
            values=log_mult_prob_truncated
        )

    log_p_y_new = sparse.BCOO(
        args=(
            jnp.concatenate(arrays=log_p_y_new['data'], axis=0),
            jnp.concatenate(arrays=log_p_y_new['indices'], axis=0)
        ),
        shape=mult_mixture['log_p_y'].shape
    )

    return dict(log_p_y=log_p_y_new, log_mult_prob=log_mult_prob_new)


def get_nn_coding_matrix(
    model: nnx.Module,
    sample_indices: jax.Array,
    cfg: DictConfig
) -> tuple[jax.Array, jax.Array]:
    """
    """
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

    # find K nearest-neighbours
    knn_indices = get_knn_indices(
        xb=features,
        num_nn=cfg.hparams.num_nearest_neighbors,
        ids=sample_indices
    )

    # calculate coding mtrices
    coding_matrix = get_batch_local_affine_coding(
        samples=features.astype(dtype=jnp.float32),
        knn_indices=knn_indices
    )

    return knn_indices, coding_matrix


@hydra.main(version_base=None, config_path='conf', config_name='conf')
def main(cfg: DictConfig) -> None:
    """Main function
    """
    # region ENVIRONMENT
    jax.config.update('jax_disable_jit', not cfg.jax.jit)
    jax.config.update('jax_platforms', cfg.jax.platform)

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)
    # endregion

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
        num_approx_classes=cfg.hparams.num_approx_classes,
        seed=cfg.training.seed,
        progress_bar_flag=cfg.data_loading.progress_bar
    )
    mult_mixture_2 = get_noisy_labels_fn(
        data_source=source_train,
        num_classes=cfg.dataset.num_classes,
        num_approx_classes=cfg.hparams.num_approx_classes,
        seed=cfg.training.seed + 1,
        progress_bar_flag=cfg.data_loading.progress_bar
    )
    # endregion

    # region EXPERIMENT TRACKING
    # create log folder if it does not exist
    logging.info(msg='Initialise directory to store logs')
    if not os.path.exists(path=cfg.experiment.logdir):
        logging.info(
            msg=f'Logging folder not found. Make a logdir at {cfg.experiment.logdir}'
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
        log_system_metrics=False
    ) as mlflow_run, \
    ocp.CheckpointManager(
        directory=os.path.join(
            os.getcwd(),
            cfg.experiment.logdir,
            cfg.experiment.name,
            mlflow_run.info.run_id
        ),
        item_names=('data_1', 'data_2'),
        options=ckpt_options
    ) as ckpt_mngr:

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
                desc='Warmup',
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
                    p_y=jnp.exp(mult_mixture_2['log_p_y']) if not hasattr(mult_mixture_2['log_p_y'], 'todense') else jnp.exp(mult_mixture_2['log_p_y'].todense()),
                    cfg=cfg
                )

                model_2, optimizer_2, loss_2 = train_cross_entropy(
                    model=model_2,
                    optimizer=optimizer_2,
                    dataloader=iter_dataloader_train_2,
                    p_y=jnp.exp(mult_mixture_1['log_p_y']) if not hasattr(mult_mixture_1['log_p_y'], 'todense') else jnp.exp(mult_mixture_1['log_p_y'].todense()),
                    cfg=cfg
                )

                mlflow.log_metrics(
                    metrics={
                        'warmup/loss_1': loss_1,
                        'warmup/loss_2': loss_2
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
            if (epoch_id + 1) % cfg.hparams.relabeling_every_n_epochs == 0 or epoch_id == 0:
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
                p_y=jnp.exp(mult_mixture_2['log_p_y']) if not hasattr(mult_mixture_2['log_p_y'], 'todense') else jnp.exp(mult_mixture_2['log_p_y'].todense()),
                cfg=cfg
            )

            model_2, optimizer_2, loss_2 = train_cross_entropy(
                model=model_1,
                optimizer=optimizer_2,
                dataloader=iter_dataloader_train_2,
                p_y=jnp.exp(mult_mixture_1['log_p_y']) if not hasattr(mult_mixture_1['log_p_y'], 'todense') else jnp.exp(mult_mixture_1['log_p_y'].todense()),
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
                    'accuracy/model_2': acc_2,
                    'accuracy/average': 0.5 * (acc_1 + acc_2)
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
