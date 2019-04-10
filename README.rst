Gated Graph Neural Network for Molecules
***********************************************

Implementation of Gated graph neural network for molecules in tensorflow.

`Gated graph neural network paper <https://arxiv.org/abs/1511.05493>`_

Dependencies
============

1. Rdkit

.. code:: shell

    conda install -c rdkit rdkit

2. Tensorflow

- cpu-version

.. code:: shell

    pip install tensorflow

- gpu-version

.. code:: shell

    pip install tensorflow-gpu

3. others(for preprocessing): pandas, tqdm, sklearn

.. code:: shell

    conda install pandas tqdm sklearn


Preprocessing
=============

1. Data
-------

`Tox21 Data <https://tripod.nih.gov/tox21/challenge/about.jsp>`_
was used for training.

sdf data was converted to graph.(Only mols valid by rdkit was used.)
graph data consists:

- node features: 

  - atomic number(onehot)
  - atomic number
  - formal charge
  - radical eletrons
  - is aromatic(bool)
  - hybridization(onehot)
  
- edge features:

  - bond type(onehot)
  - is in ring(bool)

2. preprocess.py
----------------

Does the following steps:

1. Extracts labels from csv file.

2. Check mols.

3. Get features.

4. Saves train/val/test data.

Training
========

1. Model
--------

Variant of Gated Graph Neural Network(GGNN)

- modifications:

  - modified for undirected graph
  - edge hidden features are used instead of weights per edge types
  - node hidden features are updated via stacked GRUs instead of one GRU

2. train.ipynb
--------------

Does the following steps:

1. Loads preprcessed data

2. Trains with missing labels

3. Saves weights(best)

Result
======

1. saved_models
---------------

Current trained model(weights.best.hdf5) is in this folder.

2. evaluate.ipynb
-----------------

Model can be evaluated.

ex) Receiver Operating Characteristic curve of current saved model.

.. image:: https://github.com/YunjaeChoi/ggnnmols/blob/master/doc/image/roc.png
