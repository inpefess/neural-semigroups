#!/bin/bash

set -e
PACKAGE_NAME=neural_semigroups
pycodestyle --max-doc-length 160 --ignore E501,W504 \
	    ${PACKAGE_NAME} scripts tests
pylint --rcfile=.pylintrc ${PACKAGE_NAME} scripts
mypy --config-file mypy.ini ${PACKAGE_NAME} scripts tests
pytest --cov ${PACKAGE_NAME} --cov-report term-missing --cov-fail-under=80 \
       --junit-xml test-results/neural-semigroups.xml tests/
cloc ${PACKAGE_NAME} scripts
