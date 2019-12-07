# Neural Semigroups

Here we try to learn Cayley tables of semigroups using neural
networks. The supposed workflow:

* install the package
* get the data
* build a model
* print a model testing report

## Package Installation

First, get Python 3.7

Then run several commands in a terminal:

```bash
git clone git@bitbucket.org:inpefess/neural-semigroups.git
cd neural-semigroups
python3 -m venv .
source ./bin/activate
pip install -r requirements.txt
pip install -e .
```

This could take some time (mostly downloading `torch`).

After that you can go to the project's directory and `source ./bin/activate` to
start a virtual environment there.

## Getting Data

This package uses data included into a `smallsemi` package for GAP system.
One can download it from [the GAP page](https://www.gap-system.org/pub/gap/gap4/tar.gz/packages/smallsemi-0.6.11.tar.gz).

You can get the data by running a script from a `scripts` folder:
```bash
cd scripts
./download_smallsemi.sh
```

## Training a Model

Here are several examples of commands to train a model depending on semigroup's
cardinality:

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
