# Neural Semigroups

Here we try to learn Cayley tables of semigroups using neural
networks. The supposed workflow:

* install the package
* get the data
* build a model
* print a model testing report

More documentation can be found [here](https://neural-semigroups.readthedocs.io).

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
filled Cayley table of a semigroup as a form of amuzement. As a
result, one can create a puzzle from a full Cayley table of a
semigroup but there may be many distinct solutions.

## Package Installation

First, get Python 3.8

Then run several commands in a terminal:

```bash
git clone git@bitbucket.org:inpefess/neural-semigroups.git
cd neural-semigroups
python -m venv venv
source ./venv/bin/activate
pip install -U pip poetry
poetry install
```

This could take some time (mostly downloading `torch`).

After that you can go to the project's directory and `source
./venv/bin/activate` to start a virtual environment there.

## Getting Data

This package uses data included into a `smallsemi` package for GAP
system. One can download it from [the GAP
page](https://www.gap-system.org/pub/gap/gap4/tar.gz/packages/smallsemi-0.6.11.tar.gz).

You can get the data by running a script from a `scripts` folder:
```bash
cd scripts
./download_smallsemi.sh
```

## Training a Model

Here are several examples of commands to train a model depending on
semigroup's cardinality:

```bash
python train_denoising_autoencoder.py --cardinality 4 --epochs 100 \
--learning_rate 0.1 --batch_size 32 --train_size 10 --validation_size 10
```
```bash
python train_denoising_autoencoder.py --cardinality 5 --epochs 100 \
--learning_rate 0.01 --batch_size 256 --train_size 100 --validation_size 100
```
```bash
python train_denoising_autoencoder.py --cardinality 6 --epochs 100 \
--learning_rate 0.001 --batch_size 2048 --train_size 1000 --validation_size 100
```
```bash
python train_denoising_autoencoder.py --cardinality 7 --epochs 100 \
--learning_rate 0.001 --batch_size 2048 --train_size 1000 --validation_size 100
```

## Printing a Testing Report

One can print a model quality report using the following command:

```bash
python test_model.py --cardinality 4
```
