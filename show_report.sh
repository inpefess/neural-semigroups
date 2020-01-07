#!/bin/bash

PACKAGE_NAME=neural_semigroups
pycodestyle ${PACKAGE_NAME} scripts tests
pylint --rcfile=.pylintrc ${PACKAGE_NAME} scripts
mypy ${PACKAGE_NAME} scripts tests
coverage run --source ${PACKAGE_NAME} -m unittest discover -s tests/
coverage report -m
cloc ${PACKAGE_NAME} scripts
