# Neural Semigroups

Here we try to learn Cayley tables of semigroups using neural
networks. The supposed workflow:

* install the package
* get data
* build a model
* play with the output

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

## Generating Data

```bash
cd examples
./generate_data.sh 3
```

These procedures are quite ineffective and can take hours even for `dim == 4`.
They were not tested for `dim > 4`.

## Training a Model

```bash
cd examples
./train.sh 4
```

It can take several minutes. For better training script tuning see the `train.py` source.

## Playing with Output

From a virtual environment.

```bash
cd examples
jupyter notebook
```

The web browser should run.
Navigate to `guess_tables.ipynb` Jupyter notebook and follow the instructions
there.
