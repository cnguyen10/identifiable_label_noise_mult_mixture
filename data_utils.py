import collections
import os
from typing import List, Any

from tensorflow_datasets.core import features as features_lib
from tensorflow_datasets.core.utils import type_utils
from tensorflow_datasets.core.utils import tree_utils
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf


_Example = collections.namedtuple('_Example', ['image_path', 'label'])
SplitExampleDict = dict[str, List[_Example]]


def image_folder_label_csv(
    root_dir: str,
    csv_path: str,
    sample_indices: List[int] | None = None,
    image_shape: type_utils.Shape | None = None,
    image_dtype: tf.DType | None = None
) -> tuple[tf.data.Dataset, List[str]]:
    """Generate an image folder dataset which is similar to:
    `tfds.DatasetBuilder.as_dataset(as_supervised=True)`

    Args:
        root_dir: the path to the root folder of the dataset
        csv_path: a csv format (delimiter = ' ') with two columns where the
            firt column is relative path to each image and the second column is
            the corresponding label
        sample_indices: select certain samples defined through indices in the
            dataset. If None, all samples are selected
        image_shape: a tuple consisting of the image shape HWC
        image_dtype:

    Returns:
        ds: a `tf.data.Dataset` following the format of
            `tfds.DatasetBuilder.as_dataset(as_supervised=True)`
        labels: a list of all sample labels
    """
    # extract image paths and the corresponding labels
    _split_examples, labels = _get_label_images_from_file(
        root_dir=root_dir,
        csv_path=csv_path,
        sample_indices=sample_indices,
        delimiter=' '
    )

    # define the configuration features of the dataset
    features = features_lib.FeaturesDict({
        'image': features_lib.Image(
            shape=image_shape,
            dtype=image_dtype,
        ),
        'label': features_lib.ClassLabel(),
        'image/filename': features_lib.Text(),
    })
    features['label'].names = sorted(labels)

    # Extract all labels/images
    image_paths = []
    labels = []
    examples = _split_examples
    for example in examples:
        image_paths.append(example.image_path)
        labels.append(features['label'].str2int(example.label))

    # Build the tf.data.Dataset object
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Fuse load and decode into one function
    def _load_and_decode_fn(*args, **kwargs):
        ex = _load_example(*args, **kwargs)
        return features.decode_example(ex, decoders=None)

    ds = ds.map(
        _load_and_decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # set as_supervised = True to output (image, label)
    ds = ds.map(map_func=lookup_nest)

    return ds, labels


def _load_example(
    path: tf.Tensor,
    label: tf.Tensor,
) -> dict[str, tf.Tensor]:
    img = tf.io.read_file(path)
    return {
        'image': img,
        'label': tf.cast(label, tf.int64),
        'image/filename': path,
    }


def lookup_nest(features: dict[str, Any], supervised_keys: tuple[Any, ...] = (('image', 'label'))) -> tuple[Any, ...]:
    """Converts `features` to the structure described by `supervised_keys`.

    Note that there is currently no way to access features in nested
    feature dictionaries.

    Args:
        features: dictionary of features

    Returns:
        A tuple with elements structured according to `supervised_keys`
    """
    return tree_utils.map_structure(
        lambda key: features[key], supervised_keys
    )


def _get_label_images_from_file(root_dir: str, csv_path: str, delimiter: str = ',', sample_indices: List[int] = None) -> tuple[SplitExampleDict, List[str]]:
    """Extract all label names and associated images from a csv file

    Args:
        root_dir: path to the root folder
        csv_path: path to the csv file whose each row: (image_file, label)
        where the ```image_file``` is a relative path regarding ```root_dir```

    Returns:
        split_examples: Mapping split_names -> List[_Example]
        labels: The list of labels
    """
    examples = []
    labels = set()

    with open(file=csv_path, mode='r') as f:
        for row_id, line in enumerate(f.readlines()):
            if sample_indices is not None:
                if row_id not in sample_indices:
                    # skip
                    continue

            columns = line.replace('\n', '').split(sep=delimiter)
            image_path = os.path.join(root_dir, columns[0])
            label_name = columns[1]
            examples.extend(
                [
                    _Example(image_path=image_path, label=label_name)
                ]
            )

            if label_name not in labels:
                labels.add(label_name)

    return examples, sorted(labels)
