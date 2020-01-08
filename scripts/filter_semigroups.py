"""
   Copyright 2019-2020 Boris Shminke

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from neural_semigroups.magma import Magma


def main():
    """ do all """
    parser = ArgumentParser()
    parser.add_argument(
        "--dim", type=int,
        help="magma cardinality"
    )
    choices = ["identity", "inverses"]
    parser.add_argument(
        "--filter", choices=choices,
        help="additional property"
    )
    args = parser.parse_args()
    input_file = "semigroup" if args.filter == choices[0] else "monoid"
    with open(f"./databases/{input_file}.{args.dim}.dat", "r") as file:
        cayley_tables = [
            np.array(list(map(int, line.split(" "))))
            .reshape(args.dim, args.dim)
            for line in file.readlines()
        ]
    output_file = "monoid" if args.filter == choices[0] else "group"
    with open(f"./databases/{output_file}.{args.dim}.dat", "w") as file:
        for cayley_table in tqdm(cayley_tables):
            write_to_file = False
            if args.filter == choices[0]:
                write_to_file = Magma(cayley_table).identity >= 0
            elif args.filter == choices[1]:
                write_to_file = Magma(cayley_table).has_inverses
            if write_to_file:
                file.write(str(cayley_table.reshape(-1))[1:-1] + "\n")


if __name__ == "__main__":
    main()
