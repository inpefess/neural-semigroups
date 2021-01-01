"""
   Copyright 2019-2021 Boris Shminke

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
from argparse import ArgumentParser, Namespace
from enum import Enum
from typing import List

import torch
from tqdm import tqdm

from neural_semigroups.magma import Magma


class Choices(Enum):
    """ alternatives for a menu """

    IDENTITY = "identity"
    INVERSES = "inverses"


def get_arguments(choices: List[str]) -> Namespace:
    """
    get command line arguments

    :param choices: what to filter
    :returns: an ``argparse`` namespace
    """
    parser = ArgumentParser()
    parser.add_argument("--dim", type=int, help="magma cardinality")
    parser.add_argument(
        "--filter", choices=choices, help="additional property"
    )
    return parser.parse_args()


def load_database(choice: Choices, dim: int) -> torch.Tensor:
    """
    load Cayley tables from file

    :param choice: what type of magmas to filter
    :param dim: the dimension of magma
    :returns: the database of Cayley tables
    """
    input_file = "semigroup" if choice == Choices.IDENTITY else "monoid"
    with torch.load(f"./databases/{input_file}.{dim}.zip") as torch_zip_file:
        cayley_tables = torch_zip_file["database"]
    return cayley_tables


def write_database(
    database: List[torch.Tensor], choice: Choices, dim: int
) -> None:
    """
    write Cayley tables to file

    :param database: the database of Cayley tables
    :param choice: what type of magmas to filter
    :param dim: the dimension of magma
    """
    output_file = "monoid" if choice == Choices.IDENTITY else "group"
    torch.save(
        {"database": torch.stack(database)},
        f"./databases/{output_file}.{dim}.zip",
        _use_new_zipfile_serialization=True,
    )


def main():
    """ do all """
    args = get_arguments([Choices.IDENTITY.value, Choices.INVERSES.value])
    cayley_tables = load_database(args.filter, args.dim)
    filtered = list()
    for cayley_table in tqdm(cayley_tables):
        if (
            args.filter == Choices.IDENTITY
            and Magma(cayley_table).identity >= 0
        ):
            filtered.append(cayley_table)
        elif (
            args.filter == Choices.INVERSES
            and Magma(cayley_table).has_inverses
        ):
            filtered.append(cayley_table)
    write_database(filtered, args.filter, args.dim)


if __name__ == "__main__":
    main()
