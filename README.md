[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/inpefess/neural-semigroups/blob/master/examples/train_a_model.ipynb) [![PyPI version](https://badge.fury.io/py/neural-semigroups.svg)](https://badge.fury.io/py/neural-semigroups) [![CircleCI](https://circleci.com/gh/inpefess/neural-semigroups.svg?style=svg)](https://circleci.com/gh/inpefess/neural-semigroups) [![Documentation Status](https://readthedocs.org/projects/neural-semigroups/badge/?version=latest)](https://neural-semigroups.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/inpefess/neural-semigroups/branch/master/graph/badge.svg)](https://codecov.io/gh/inpefess/neural-semigroups)

# Neural Semigroups

Here we try to model Cayley tables of semigroups using neural networks.

The simplest way to get started is to [use Google Colaboratory](https://colab.research.google.com/github/inpefess/neural-semigroups/blob/master/examples/train_a_model.ipynb)

More documentation can be found 
[here](https://neural-semigroups.readthedocs.io).

## Motivation

This work was inspired by [a sudoku
solver](https://github.com/Kyubyong/sudoku). A solved Sudoku puzzle
is nothing more than a Cayley table of a quasigroup from 9 items with
some well-known additional properties. So, one can imagine a puzzle
made from a Cayley table of any other magma, e. g. a semigroup, by
hiding part of its cells.

There are two major differences between sudoku and puzzles based on
semigroups:

1) it's easy to take a glance on a table to understand whether it is
a sudoku or not. That's why it was possible to encode numbers in a
table cells as colour intensities. Sudoku is a picture, and a
semigroup is not. It's difficult to check a Cayley table's
associativity with a naked eye;

2) sudoku puzzles are solved by humans for fun and thus catalogued.
When solving a sudoku one knows for sure that there is a unique
solution. On the contrary, nobody guesses values in a partially
filled Cayley table of a semigroup as a form of amusement. As a
result, one can create a puzzle from a full Cayley table of a
semigroup but there may be many distinct solutions.
