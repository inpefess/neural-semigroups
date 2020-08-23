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
import os
from argparse import ArgumentParser
from itertools import chain
from multiprocessing import Pool
from typing import List, Tuple

import torch
from tqdm import tqdm

from neural_semigroups.magma import Magma
from neural_semigroups.utils import get_magma_by_index


def find_semigroups(arg) -> List[torch.Tensor]:
    """ searches for semigroups iteratively going through magmas

    :param arg: a tuple (
        a magma from which to start searching,
        a number of next magmas to look through
    )
    :returns: a list of semigroups' Cayley tables
    """
    magma: Magma = arg[0]
    batch_size: int = arg[1]
    semigroups = list()
    if magma.is_associative:
        semigroups.append(magma.cayley_table)
    for _ in tqdm(range(batch_size), desc="searching semigroups in a batch"):
        magma = magma.next_magma
        if magma.is_associative:
            semigroups.append(magma.cayley_table)
    return semigroups


def get_starting_magmas(
    batch_count: int, cardinality: int
) -> Tuple[List[Magma], List[int]]:
    """
    generates a list of starting points of parallel search

    :param batch_count: how many magmas to generate
    :param cardinality: a number of elements in each magma
    :returns: a pair (list of the starting magmas, list of batch sizes)
    """
    magma_count = cardinality ** (cardinality ** 2)
    batch_sizes = batch_count * [magma_count // batch_count]
    batch_sizes[-1] += magma_count - 1 - batch_sizes[0] * batch_count
    starting_magmas = []
    index = 0
    for batch_number in range(batch_count):
        starting_magmas.append(get_magma_by_index(cardinality, index))
        index += batch_sizes[batch_number]
    return starting_magmas, batch_sizes


def main() -> None:
    """ do all """
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--dim", type=int, help="magma cardinality")
    args = argument_parser.parse_args()
    cpu_count = os.cpu_count()
    batch_count = cpu_count - 1 if cpu_count is not None else 1
    starting_magmas, batch_sizes = get_starting_magmas(batch_count, args.dim)
    with Pool(batch_count) as pool:
        semigroups_lists = pool.map(
            find_semigroups, zip(starting_magmas, batch_sizes)
        )
    torch.save(
        {"database": torch.stack(list(chain.from_iterable(semigroups_lists)))},
        f"./databases/semigroup.{args.dim}.zip",
        _use_new_zipfile_serialization=True,
    )


if __name__ == "__main__":
    main()
