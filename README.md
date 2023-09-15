This repository contains the code development for the paper [Towards the Identifiability in Noisy Label Learning: A Multinomial Mixture Approach](https://arxiv.org/abs/2301.01405).

[**Requirements**](#python-package-requirements)
| [**Dataset structure**](#dataset-structure)
| [**Experiment tracking**](#experiment-tracking)

## Python package requirements
The code mainly relies on JAX - a composable machine learning framework developed by Google. In particular, the implementation contains the following packages:
- ```jax``` - a composable machine learning framework
- ```flax``` - neural networks with JAX
- ```tensorflow``` - an end-to-end machine learning platform
- ```tensorflow-datasets``` (a.k.a. ```tfds```) - high-level wrapper for ```tensorflow.data```
- ```tensorflow-probability``` - a library for probabilistic reasoning and statistical analysis
- ```faiss-gpu``` - similarity search developed by Facebook AI
- ```aim``` - a library for experiment tracking,

and many other packages. Further details can be referred to the [requirements.txt](requirements.txt) included in this repository. Please follow the instructions of JAX and TensorFlow to install those packages since their CPU and GPU versions might be different.

## Dataset structure
Each of the datasets used in the implemetation is characterised by a ```csv``` file which have a similar structure as in ```Red/Blue mini-ImageNet```. Such a ```csv``` file has two coumns:
```
relative/path/of/image1.ext 8
relative/path/of/image2.ext 6
...
```
where the first column is the relative path of a sample and the second column is its corresponding noisy label.

## Experiment tracking
Experiments are tracked and managed by ```aim```. To see the result, open a terminal and run the following command:
```linux
aim up --repo logs
```