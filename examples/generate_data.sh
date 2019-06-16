#!/bin/bash

dim=$1
source ../bin/activate
python generate_semigroups.py --dim ${dim}
python filter_semigroups.py --dim ${dim} --filter identity
python filter_semigroups.py --dim ${dim} --filter inverses
deactivate
