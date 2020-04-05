.. _for-developers:

Installing Neural Semigroups for Development
============================================

First, get Python 3.6

Then run several commands in a terminal: ::

  git clone git@github.com:inpefess/neural-semigroups.git
  cd neural-semigroups
  python -m venv venv
  source ./venv/bin/activate
  pip install -U pip poetry
  poetry install

This could take some time (mostly downloading ``torch``).

After that you can go to the project's directory and ``source ./venv/bin/activate`` to start a virtual environment there.
