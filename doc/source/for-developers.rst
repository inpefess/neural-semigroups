.. _for-developers:

Installing Neural Semigroups for Development
============================================

First, get Python 3.6 or later (for now, it doesn't support Python 3.9 because of ``torch`` dependency).

Then run several commands in a terminal: ::

  git clone git@github.com:inpefess/neural-semigroups.git
  cd neural-semigroups
  python -m venv venv
  source ./venv/bin/activate
  pip install -U pip poetry wheel setuptools
  poetry install

This could take some time (mostly downloading ``torch``).

After that you can go to the project's directory and ``source ./venv/bin/activate`` to start a virtual environment there.

It's also recommended to run ``pre-commit install`` from a repo root folder to get a pre-commit hook initialized.

To run code check similar to what is going on at ``circleci`` simply ``./show_report.sh`` from the project's root.
