.. _getting-started:

Getting Started with Neural Semigroups
======================================

The simplest way is to run on `Google Colaboratory`_.

First, install the package: ::

  pip install neural-semigroups

Second, download and run `an example`_.

If you're looking for something more advanced see :ref:`this page<for-developers>`.

More about data
---------------

This package uses data included into a ``smallsemi`` package for GAP system. One can download it from `the GAP page`_.

Data is downloaded and saved automatically. By default it resides in a directory ``$HOME/neural-semigroups-data/smallsemi-data/``. Ona can change this using a ``data_path`` argument of a :ref:`Cayley database<cayley-database>`.

To get some data, just: ::

  from neural_semigroups import CayleyDatabase

  cayley_db = CayleyDatabase(4)


.. _the GAP page: https://www.gap-system.org/pub/gap/gap4/tar.gz/packages/smallsemi-0.6.12.tar.gz
.. _an example: https://github.com/inpefess/neural-semigroups/blob/master/examples/train_a_model.ipynb
.. _Google Colaboratory: https://colab.research.google.com/
