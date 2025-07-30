This repository contains the code development for the paper [Towards the Identifiability in Noisy Label Learning: A Multinomial Mixture Approach](https://arxiv.org/abs/2301.01405).

[**Requirements**](#python-package-requirements)
| [**Dataset structure**](#dataset-structure)
| [**Experiment tracking**](#experiment-tracking)

## Python package requirements
The code mainly relies on JAX - a composable machine learning framework developed by Google. In particular, the implementation contains the following packages:
- `jax` - a composable machine learning framework
- `flax` - neural networks with JAX
- `grain` - a framework agnostic data loading
- `albumentations` - a library for image augmentations (similar to `torchvision`)
- `tensorflow-probability` - a library for probabilistic reasoning and statistical analysis
- `faiss-gpu` - similarity search developed by Facebook AI
- `jaxopt` - an optimisation in JAX
- `hydra-core` - a library for configuration
- `mlflow` - a library manage and track experiments.

Further details can be referred to the [requirements.txt](requirements.txt) included in this repository. Please follow the instructions of JAX and TensorFlow to install those packages since their CPU and GPU versions might be different.

**Do not install packages directly from `requirements.txt`**
- For `jax`, please follow the instruction in its [installation guide](https://docs.jax.dev/en/latest/installation.html) to install the GPU version
- For `faiss-gpu`, it is stopped to be published on https://pypi.org/project/. Please either compile from source or download its prebuilt wheel at https://github.com/kyamagu/faiss-wheels/releases/tag/v1.7.3 and install locally. **Note:** after installing `faiss-gpu`, please downgrade `numpy` to the version specified in `requirements.txt` because this version of `faiss-gpu` is pinned to `numpy` 1.x.

## Dataset structure
A dataset has the following folder structure:
```bash
[
    {
        "file": "<path_to_one_image_file>",
        "label": [2, 9, 0]
    },
]
```

Each of the datasets used in the implemetation is characterised by a `json` file.

## Experiment tracking
Experiments are tracked and managed in `mlflow`. To see the result, open a terminal and run the following command:
```bash
bash mlflow_server.sh
```