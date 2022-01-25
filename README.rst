|Open In Colab| |PyPI version| |CircleCI| |Documentation Status|
|codecov|

Neural Semigroups
=================

The project is abandoned.

If you want to reproduce results from the
`paper <https://arxiv.org/abs/2103.07388>`__, please use `this
notebook <https://colab.research.google.com/github/inpefess/neural-semigroups/blob/master/examples/train_a_model.ipynb>`__.

Here we try to model Cayley tables of semigroups using neural networks.

This work was inspired by `a sudoku
solver <https://github.com/Kyubyong/sudoku>`__. A solved Sudoku puzzle
is nothing more than a Cayley table of a quasigroup from 9 items with
some well-known additional properties. So, one can imagine a puzzle made
from a Cayley table of any other magma, e.g. a semigroup, by hiding part
of its cells.

There are two major differences between sudoku and puzzles based on
semigroups:

1) it’s easy to take a glance on a table to understand whether it is a
   sudoku or not. That’s why it was possible to encode numbers in a
   table cells as colour intensities. Sudoku is a picture, and a
   semigroup is not. It’s difficult to check a Cayley table’s
   associativity with a naked eye;

2) Sudoku puzzles are solved by humans for fun and thus catalogued. When
   solving a sudoku one knows for sure that there is a unique solution.
   On the contrary, nobody guesses values in a partially filled Cayley
   table of a semigroup as a form of amusement. As a result, one can
   create a puzzle from a full Cayley table of a semigroup but there may
   be many distinct solutions.

How to Install
==============

The best way to install this package is to use ``pip``:

.. code:: sh

   pip install neural-semigroups

How to use
==========

The simplest way to get started is to `use Google
Colaboratory <https://colab.research.google.com/github/inpefess/neural-semigroups/blob/master/examples/dae_4_colab.ipynb>`__.

To look at more experimental results for different semigroups
cardinalities, you can use `Kaggle <https://kaggle.com>`__:

-  `cardinality
   4 <https://www.kaggle.com/inpefess/neural-semigroups-dae-dim-4>`__
-  `cardinality
   5 <https://www.kaggle.com/inpefess/neural-semigroups-dae-dim-5>`__

There is also an experimental
`notebook <https://github.com/inpefess/neural-semigroups/blob/master/examples/ExperimentNotebook.ipynb>`__
contributed by `Žarko Bulić <https://github.com/zarebulic>`__.

How to Contribute
=================

`Pull requests <https://github.com/inpefess/neural-semigroups/pulls>`__
are welcome. To start:

.. code:: sh

   git clone https://github.com/inpefess/neural-semigroups
   cd neural-semigroups
   # activate python virtual environment with Python 3.6+
   pip install -U pip
   pip install -U setuptools wheel poetry
   poetry install
   # recommended but not necessary
   pre-commit install

To check the code quality before creating a pull request, one might run
the script
`show_report.sh <https://colab.research.google.com/github/inpefess/neural-semigroups/blob/master/show_report.sh>`__.
It locally does nearly the same as the CI pipeline after the PR is
created.

Reporting issues or problems with the software
==============================================

Questions and bug reports are welcome on `the
tracker <https://github.com/inpefess/neural-semigroups/issues>`__.

More documentation
==================

More documentation can be found
`here <https://neural-semigroups.readthedocs.io/en/latest>`__.

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/inpefess/neural-semigroups/blob/master/examples/dae_4_colab.ipynb
.. |PyPI version| image:: https://badge.fury.io/py/neural-semigroups.svg
   :target: https://badge.fury.io/py/neural-semigroups
.. |CircleCI| image:: https://circleci.com/gh/inpefess/neural-semigroups.svg?style=svg
   :target: https://circleci.com/gh/inpefess/neural-semigroups
.. |Documentation Status| image:: https://readthedocs.org/projects/neural-semigroups/badge/?version=latest
   :target: https://neural-semigroups.readthedocs.io/en/latest/?badge=latest
.. |codecov| image:: https://codecov.io/gh/inpefess/neural-semigroups/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/inpefess/neural-semigroups
