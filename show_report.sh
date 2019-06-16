#!/bin/bash

PACKAGE_NAME=neural_semigroups
pycodestyle ${PACKAGE_NAME} examples tests
pylint --rcfile=.pylintrc ${PACKAGE_NAME} examples
mypy ${PACKAGE_NAME} examples tests
coverage run --source ${PACKAGE_NAME} -m unittest discover -s tests/
coverage report -m
cloc ${PACKAGE_NAME} examples
