import jax
from jax import numpy as jnp

from flax.training import train_state
from flax.training import orbax_utils
import orbax.checkpoint as ocp

import optax


import tensorflow_datasets as tfds
import tensorflow as tf

import numpy as np

import aim

import random

import subprocess
import os
from pathlib import Path

from tqdm import tqdm
import logging
import argparse
import chex

from PreactResnet import ResNet18
from utils import parse_arguments, get_features, get_knn_indices, get_p_y, \
    get_batch_local_affine_coding, train_model, TrainState
from data_utils import image_folder_label_csv


def relabel_data(
        log_p_y: chex.Array,
        log_mult_prob: chex.Array,
        nn_idx: chex.Array,
        coding_matrix: chex.Array,
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

    Returns:
        p_y: a sparse matrix where each row is p(y | x)  # (N, C)
        mult_prob: a sparse 3-d tensor where each matrix is p(yhat | x, y)  # (N, C , C)
    """
    num_samples = log_p_y.shape[0]

    # initialize new p(y | x) and p(yhat | x, y)
    log_p_y_new = log_p_y + 0.
    log_mult_prob_new = log_mult_prob + 0.

    data_loader = tf.data.Dataset.from_tensor_slices(
        tensors=jnp.arange(start=0, stop=num_samples, step=1)
    )
    data_loader = data_loader.batch(batch_size=args.batch_size_em)

    for indices in tqdm(iterable=tfds.as_numpy(dataset=data_loader), desc=' re-label', leave=False, position=1):
        # extract the corresponding noisy labels
        log_p_y_ = log_p_y[indices]  # (B, C)
        log_mult_prob_ = log_mult_prob[indices]  # (B, C, C)

        # extract coding coefficients
        log_nn_coding = jnp.log(coding_matrix[indices])  # (B, K)

        # region APPROXIMATE p(y | x) through nearest-neighbours
        nn_idx_ = nn_idx[indices]
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

        # if jnp.any(a=jnp.isnan(log_p_y_temp)):
        #     raise ValueError('NaN is detected after running EM')

        # update the noisy labels
        log_p_y_new = log_p_y_new.at[indices].set(values=log_p_y_temp)
        log_mult_prob_new = log_mult_prob_new.at[indices].set(values=log_mult_prob_temp)

    return log_p_y_new, log_mult_prob_new


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
    tf.config.experimental.set_visible_devices([], 'GPU')  # disable GPU

    # # limit GPU memory for TensorFlow
    # gpus = tf.config.list_physical_devices('GPU')
    # tf.config.set_logical_device_configuration(
    #     device=gpus[0],
    #     logical_devices=[tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
    # )
    # endregion

    # region DATASET
    # convert image-shape from string to int
    args.img_shape = tuple([int(ishape) for ishape in args.img_shape])

    # load training dataset
    log_p_y_1 = 0.
    for label_filename in args.label_filenames:
        dataset_train, labels = image_folder_label_csv(
            root_dir=os.path.join(args.dataset_root, 'train'),
            csv_path=os.path.join(args.dataset_root, 'split', label_filename),
            sample_indices=None,
            image_shape=args.img_shape
        )

        args.num_classes = len(set(labels))
        args.total_num_samples = len(dataset_train)

        # make a dataset of noisy labels
        label_dataset = tf.data.Dataset.from_tensor_slices(tensors=labels).batch(batch_size=args.batch_size)

        log_p_y_1_temp = jnp.empty(shape=(args.total_num_samples, args.num_classes), dtype=jnp.float32)  # (N, C)
        current_index = 0
        for yhat in tqdm(iterable=tfds.as_numpy(dataset=label_dataset), desc='get noisy labels', leave=False, position=1):
            # track how many samples
            batch_size_ = yhat.shape[0]

            # convert integer numbers to one-hot vectors
            noisy_labels = jax.nn.one_hot(x=yhat, num_classes=args.num_classes, dtype=jnp.float32)

            # assign to yhats
            log_p_y_1_temp = log_p_y_1_temp.at[current_index:(current_index + batch_size_)].set(noisy_labels)

            current_index = current_index + batch_size_

        log_p_y_1 = log_p_y_1 + log_p_y_1_temp

    log_p_y_1 = log_p_y_1 / len(args.label_filenames)

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

    del labels
    del label_dataset
    del current_index
    del batch_size_
    # endregion

    # region MODEL
    model = ResNet18(num_classes=args.num_classes)

    args.lr_schedule_fn = optax.cosine_decay_schedule(
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
        tx = optax.sgd(learning_rate=args.lr_schedule_fn, momentum=0.9)  # define an optimizer

        state = TrainState.create(
            apply_fn=model.apply,
            params=params['params'],
            batch_stats=params['batch_stats'],
            model=model,
            model_id=model_id,
            tx=tx
        )

        return state

    state_1, state_2 = [initialize_train_state(model_id=model_id) for model_id in range(2)]
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
        run_hash=args.run_hash_id,
        repo=args.logdir,
        read_only=False,
        experiment=args.experiment_name,
        force_resume=False,
        capture_terminal_logs=False,
        system_tracking_interval=600  # capture every x seconds
    )
    aim_run['hparams'] = {key: args.__dict__[key] for key in args.__dict__ if isinstance(args.__dict__[key], (int, bool, str, float))}

    # create a folder with the corresponding hash run id to store checkpoints
    args.checkpoint_dir = os.path.join(args.logdir, aim_run.hash)
    if not os.path.exists(path=args.checkpoint_dir):
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # endregion

    # region SETUP CHECKPOINT and RESTORE
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=1,
        save_interval_steps=1
    )
    checkpoint_mngr = ocp.CheckpointManager(
        directory=args.checkpoint_dir,
        checkpointers={
            'state_1': ocp.PyTreeCheckpointer(),
            'state_2': ocp.PyTreeCheckpointer()
        },
        options=checkpoint_options
    )
    # endregion

    if args.resume:
        # must be associated with a run hash id
        assert args.run_hash_id is not None

        # default restore will be in raw dictionary
        # create an example structure to restore to the dataclass of interest
        checkpoint_example = {
            'state_1': {
                'state': state_1,
                'log_p_y': log_p_y_1,
                'log_mult_prob': log_mult_prob_1
            },
            'state_2': {
                'state': state_2,
                'log_p_y': log_p_y_2,
                'log_mult_prob': log_mult_prob_2
            }
        }

        restored = checkpoint_mngr.restore(
            step=checkpoint_mngr.latest_step(),
            items=checkpoint_example
        )

        state_1 = restored['state_1']['state']
        log_p_y_1 = jnp.asarray(a=restored['state_1']['log_p_y'])
        log_mult_prob_1 = jnp.asarray(a=restored['state_1']['log_mult_prob'])

        state_2 = restored['state_2']['state']
        log_p_y_2 = jnp.asarray(restored['state_2']['log_p_y'])
        log_mult_prob_2 = jnp.asarray(a=restored['state_2']['log_mult_prob'])

        del checkpoint_example
        del restored

        # somehow model cannot be restored
        # re-initialize the model
        state_1, state_2 = [TrainState.create(
            apply_fn=model.apply,
            params=state_.params,
            batch_stats=state_.batch_stats,
            model=model,
            model_id=state_.model_id.item(),
            tx=state_.tx
        ) for state_ in [state_1, state_2]]
    else:
        # region WARM-UP
        for state_ in [state_1, state_2]:
            state_ = train_model(
                state=state_,
                dataset_train=dataset_train,
                p_y=jnp.exp(log_p_y_1),
                num_epochs=args.num_warmup,
                aim_run=aim_run,
                args=args
            )
        # endregion

    # define data-loaders to create sample indices
    def create_index_loader() -> tf.data.Dataset:
        index_loader = tf.data.Dataset.range(args.total_num_samples)
        index_loader = index_loader.shuffle(buffer_size=args.total_num_samples, reshuffle_each_iteration=True)
        index_loader = index_loader.batch(batch_size=args.num_samples)

        return index_loader

    index_loader_1, index_loader_2 = [create_index_loader() for _ in range(2)]

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
                    sub_dataset, _ = image_folder_label_csv(
                        root_dir=os.path.join(args.dataset_root, 'train'),
                        csv_path=os.path.join(args.dataset_root, 'split', args.label_filenames[0]),
                        sample_indices=sample_indices_,
                        image_shape=args.img_shape
                    )
                    # extract features
                    features = get_features(
                        state=state_,
                        ds=sub_dataset,
                        batch_size=args.batch_size
                    )
                    features = np.asanyarray(a=features)

                    # find K nearest-neighbours
                    knn_indices_ = get_knn_indices(xb=features, num_nn=args.k)

                    # calculate coding mtrices
                    coding_matrix_ = get_batch_local_affine_coding(samples=features, knn_indices=knn_indices_)

                    # run EM and re-label samples
                    return relabel_data(
                        log_p_y=log_p_y_,
                        log_mult_prob=log_mult_prob_,
                        nn_idx=knn_indices_,
                        coding_matrix=coding_matrix_,
                        args=args
                    )

                log_p_y_1_temp, log_mult_prob_1_temp = relabel_data_wrapper(
                    state_=state_1,
                    sample_indices_=sample_indices_1,
                    log_p_y_=log_p_y_1[sample_indices_1],
                    log_mult_prob_=log_mult_prob_1[sample_indices_1]
                )
                log_p_y_1 = log_p_y_1.at[sample_indices_1].set(values=log_p_y_1_temp)
                log_mult_prob_1 = log_mult_prob_1.at[sample_indices_1].set(values=log_mult_prob_1_temp)

                log_p_y_2_temp, log_mult_prob_2_temp = relabel_data_wrapper(
                    state_=state_2,
                    sample_indices_=sample_indices_2,
                    log_p_y_=log_p_y_2[sample_indices_2],
                    log_mult_prob_=log_mult_prob_2[sample_indices_2]
                )
                log_p_y_2 = log_p_y_1.at[sample_indices_2].set(values=log_p_y_2_temp)
                log_mult_prob_2 = log_mult_prob_2.at[sample_indices_2].set(values=log_mult_prob_2_temp)

            # region TRAIN models on relabelled data
            state_1 = train_model(
                state=state_1,
                dataset_train=dataset_train,
                p_y=jnp.exp(log_p_y_2),
                num_epochs=5,
                aim_run=aim_run,
                args=args
            )
            state_2 = train_model(
                state=state_2,
                dataset_train=dataset_train,
                p_y=jnp.exp(log_p_y_1),
                num_epochs=5,
                aim_run=aim_run,
                args=args
            )
            # endregion

            # save checkpoint
            checkpoint = {
                'state_1': {
                    'state': state_1,
                    'log_p_y': log_p_y_1,
                    'log_mult_prob': log_mult_prob_1
                },
                'state_2': {
                    'state': state_2,
                    'log_p_y': log_p_y_2,
                    'log_mult_prob': log_mult_prob_2
                }
            }
            checkpoint_mngr.save(
                step=epoch_id + 1,
                items=checkpoint,
                save_kwargs={'save_args': orbax_utils.save_args_from_target(checkpoint)}
            )
            del checkpoint

        logging.info(msg='Training is completed.')
    finally:
        aim_run.close()
        logging.info(msg='AIM is closed.\nProgram is terminated.')


if __name__ == '__main__':
    logger_current = logging.getLogger(name=__name__)
    logger_current.setLevel(level=logging.INFO)
    main()
