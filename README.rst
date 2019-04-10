Gated Graph Neural Network for Molecules
***********************************************

Gated graph neural network for molecules in tensorflow.

Dependencies
============

1. Rdkit

.. code:: shell

    conda install -c rdkit rdkit

2. Tensorflow

cpu-version

.. code:: shell

    pip install tensorflow

gpu-version

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
        1. atomic number(onehot)
	2. atomic number
	3. formal charge
	4. radical eletrons
	5. is aromatic(bool)
	6. hybridization(onehot)
    - edge features:
	1. bond type(onehot)
	2. is in ring(bool)

2. preprocess.py
----------------

Does the following steps:

1. Extracts labels from csv file.

2. Check mols.

3. Get features.

4. Saves train/val/test data.



