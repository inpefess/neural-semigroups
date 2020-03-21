Getting Started with Neural Semigroups
======================================

Getting Data
------------

This package uses data included into a ``smallsemi`` package for GAP system. One can download it from `the GAP page`_.

You can get the data by running a script from a ``scripts`` folder:::

  cd scripts
  ./download_smallsemi.sh

Training a Model
----------------

Here are several examples of commands to train a model depending on semigroup's cardinality:::

  python train_denoising_autoencoder.py --cardinality 4 --epochs 100 \
  --learning_rate 0.1 --batch_size 32 --train_size 10 --validation_size 10

  python train_denoising_autoencoder.py --cardinality 5 --epochs 100 \
  --learning_rate 0.01 --batch_size 256 --train_size 100 --validation_size 100

  python train_denoising_autoencoder.py --cardinality 6 --epochs 100 \
  --learning_rate 0.001 --batch_size 2048 --train_size 1000 --validation_size 100

  python train_denoising_autoencoder.py --cardinality 7 --epochs 100 \
  --learning_rate 0.001 --batch_size 2048 --train_size 1000 --validation_size 100

Printing a Testing Report
-------------------------

One can print a model quality report using the following command:::

  python test_model.py --cardinality 4


.. _the GAP page: https://www.gap-system.org/pub/gap/gap4/tar.gz/packages/smallsemi-0.6.12.tar.gz
