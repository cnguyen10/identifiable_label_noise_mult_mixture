"""Image Classification Folder datasets with image paths and their corresponding labels in a csv file"""

from __future__ import annotations

import collections
import os
import random
from typing import Dict, List, NoReturn, Optional, Tuple

from tensorflow_datasets.core import dataset_builder
from tensorflow_datasets.core import dataset_info
from tensorflow_datasets.core import decode
from tensorflow_datasets.core import features as features_lib
from tensorflow_datasets.core import splits as split_lib
from tensorflow_datasets.core.utils import type_utils
from tensorflow_datasets.core.utils import version
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf

_SUPPORTED_IMAGE_FORMAT = ('.jpg', '.jpeg', '.png')

_Example = collections.namedtuple('_Example', ['image_path', 'label'])

# Dict of 'split_name'-> `List[_Example]`
SplitExampleDict = Dict[str, List[_Example]]


class ImageFolder_2(dataset_builder.DatasetBuilder):
    """Generic image classification dataset created from manual directory.

    `ImageFolder` creates a `tf.data.Dataset` reading the original image files.

    The data directory should have the following structure:

    ```
    path/to/image_dir/
    split_name/  # Ex: 'train'
        label1/  # Ex: 'airplane' or '0015'
        xxx.png
        xxy.png
        xxz.png
        label2/
        xxx.png
        xxy.png
        xxz.png
    split_name/  # Ex: 'test'
        ...
    ```

    To use it:

    ```
    builder = tfds.ImageFolder('path/to/image_dir/')
    print(builder.info)  # num examples, labels... are automatically calculated
    ds = builder.as_dataset(split='train', shuffle_files=True)
    tfds.show_examples(ds, builder.info)
    ```
    """

    VERSION = version.Version('1.0.0')

    def __init__(
        self,
        root_dir: str,
        noise_rate: float,
        split_name: str,
        *,
        sample_indices: List[int] | None = None,
        shape: Optional[type_utils.Shape] = None,
        dtype: Optional[tf.DType] = None,
    ):
        """Construct the `DatasetBuilder`.

        Args:
            root_dir: Path to the directory containing the images.
            shape: Image shape forwarded to `tfds.features.Image`.
            dtype: Image dtype forwarded to `tfds.features.Image`.
        """
        self._image_shape = shape
        self._image_dtype = dtype
        super(ImageFolder_2, self).__init__()
        self._data_dir = root_dir  # Set data_dir to the existing dir.

        # Extract the splits, examples, labels
        root_dir = os.path.expanduser(root_dir)

        if noise_rate is None:
            self._split_examples, labels = _get_split_label_images(root_dir)
        else:
            self._split_examples, labels = _get_labelled_images(
                root_dir=root_dir,
                csv_path=os.path.join(root_dir, 'split', 'label_noise_{0:.1f}'.format(noise_rate)),
                split_name=split_name,
                sample_indices=sample_indices,
                delimiter=' '
            )

        # _split_examples = [_Example(image_path='/path/to/file.ext', label='dog')]
        # labels = ['dog', 'kangaroo']

        # Update DatasetInfo labels
        self.info.features['label'].names = sorted(labels)

        # Update DatasetInfo splits
        split_infos = [
            split_lib.SplitInfo(  # pylint: disable=g-complex-comprehension
                name=split_name,
                shard_lengths=[len(examples)],
                num_bytes=0,
            )
            for split_name, examples in self._split_examples.items()
        ]
        split_dict = split_lib.SplitDict(split_infos)
        self.info.set_splits(split_dict)

    def _info(self) -> dataset_info.DatasetInfo:
        return dataset_info.DatasetInfo(
            builder=self,
            description='Generic image classification dataset.',
            features=features_lib.FeaturesDict({
                'image': features_lib.Image(
                    shape=self._image_shape,
                    dtype=self._image_dtype,
                ),
                'label': features_lib.ClassLabel(),
                'image/filename': features_lib.Text(),
            }),
            supervised_keys=('image', 'label'),
        )

    def _download_and_prepare(self, **kwargs) -> NoReturn:  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        raise NotImplementedError(
            'No need to call download_and_prepare function for {}.'.format(
                type(self).__name__
            )
        )

    def download_and_prepare(self, **kwargs):  # -> NoReturn:
        return self._download_and_prepare()

    def _as_dataset(
        self,
        split: str,
        shuffle_files: bool = False,
        decoders: Optional[Dict[str, decode.Decoder]] = None,
        read_config=None,
    ) -> tf.data.Dataset:
        """Generate dataset for given split."""
        del read_config  # Unused (automatically created in `DatasetBuilder`)

        if split not in self.info.splits.keys():
            raise ValueError(
                'Unrecognized split {}. Subsplit API not yet supported for {}. '
                'Split name should be one of {}.'.format(
                    split, type(self).__name__, list(self.info.splits.keys())
                )
            )

        # Extract all labels/images
        image_paths = []
        labels = []
        examples = self._split_examples[split]
        for example in examples:
            image_paths.append(example.image_path)
            labels.append(self.info.features['label'].str2int(example.label))

        # Build the tf.data.Dataset object
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        if shuffle_files:
            ds = ds.shuffle(len(examples))

        # Fuse load and decode into one function
        def _load_and_decode_fn(*args, **kwargs):
            ex = _load_example(*args, **kwargs)
            return self.info.features.decode_example(ex, decoders=decoders)

        ds = ds.map(
            _load_and_decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return ds


def _load_example(
    path: tf.Tensor,
    label: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    img = tf.io.read_file(path)
    return {
        'image': img,
        'label': tf.cast(label, tf.int64),
        'image/filename': path,
    }


def _get_labelled_images(root_dir: str, csv_path: str, delimiter: str = ',', split_name: str = 'train', sample_indices: List[int] = None) -> Tuple[SplitExampleDict, List[str]]:
    """Extract all label names and associated images from a csv file

    Args:
        csv_path: path to the csv file whose each row: (image_file, label)

    Returns:
        split_examples: Mapping split_names -> List[_Example]
        labels: The list of labels
    """
    split_examples = collections.defaultdict(list)
    labels = set()

    with open(file=csv_path, mode='r') as f:
        for row_id, line in enumerate(f.readlines()):
            if sample_indices is not None:
                if row_id not in sample_indices:
                    # skip
                    continue

            columns = line.replace('\n', '').split(sep=delimiter)
            image_path = os.path.join(root_dir, split_name, columns[0])
            label_name = columns[1]
            split_examples[split_name].extend(
                [
                    _Example(image_path=image_path, label=label_name)
                ]
            )

            if label_name not in labels:
                labels.add(label_name)

    return split_examples, sorted(labels)


def _get_split_label_images(root_dir: str) -> Tuple[SplitExampleDict, List[str]]:
    """Extract all label names and associated images.

    This function guarantee that examples are deterministically shuffled
    and labels are sorted.

    Args:
    root_dir: The folder where the `split/label/image.png` are located

    Returns:
    split_examples: Mapping split_names -> List[_Example]
    labels: The list of labels
    """
    split_examples = collections.defaultdict(list)
    labels = set()
    for split_name in sorted(_list_folders(root_dir)):
        split_dir = os.path.join(root_dir, split_name)
        for label_name in sorted(_list_folders(split_dir)):
            labels.add(label_name)
            split_examples[split_name].extend(
                [
                    _Example(image_path=image_path, label=label_name)
                    for image_path in sorted(
                        _list_img_paths(os.path.join(split_dir, label_name))
                    )
                ]
            )

    # Shuffle the images deterministically
    for split_name, examples in split_examples.items():
        rgn = random.Random(split_name)  # Uses different seed for each split
        rgn.shuffle(examples)
    return split_examples, sorted(labels)


def _list_folders(root_dir: str) -> List[str]:
    return [
        f
        for f in tf.io.gfile.listdir(root_dir)
        if tf.io.gfile.isdir(os.path.join(root_dir, f))
    ]


def _list_img_paths(root_dir: str) -> List[str]:
    return [
        os.path.join(root_dir, f)
        for f in tf.io.gfile.listdir(root_dir)
        if any(f.lower().endswith(ext) for ext in _SUPPORTED_IMAGE_FORMAT)
    ]
