"""
   Copyright 2019 Boris Shminke

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
from multiprocessing import Pool
from subprocess import check_output
from typing import List, Tuple

from tqdm import tqdm

from neural_semigroups.magma import Magma
from neural_semigroups.utils import get_magma_by_index, next_magma


def find_semigroups(arg) -> None:
    """ searches for semigroups iteratively going through magmas

    :param arg: a tripe (
        an ordinal number of the process,
        a magma from which to start searching,
        a number of next magmas to look through
    )
    :returns: writes to the file
    """
    i: int = arg[0]
    magma: Magma = arg[1]
    batch_size: int = arg[2]
    with open(f"data{i}.dat", "w") as data_file:
        if magma.is_associative:
            line = str(magma.cayley_table.reshape(-1))
            data_file.write(line[1:-1] + "\n")
        for _ in tqdm(range(batch_size)):
            magma = next_magma(magma)
            if magma.is_associative:
                line = str(magma.cayley_table.reshape(-1))
                data_file.write(line[1:-1] + "\n")


def get_starting_magmas(
        batch_count: int,
        cardinality: int
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
    parser = ArgumentParser()
    parser.add_argument(
        "--dim", type=int,
        help="magma cardinality"
    )
    args = parser.parse_args()
    cpu_count = os.cpu_count()
    if cpu_count:
        batch_count = cpu_count - 1
    else:
        batch_count = 1
    starting_magmas, batch_sizes = get_starting_magmas(batch_count, args.dim)
    check_output(["rm -f ./data*.dat"], shell=True)
    with Pool(batch_count) as pool:
        tqdm(
            pool.map(
                find_semigroups,
                zip(
                    range(len(starting_magmas)),
                    starting_magmas,
                    batch_sizes
                )
            ),
            total=batch_count
        )
    check_output(
        [f"""
            cat ./data*.dat | sort | uniq > \
                ./databases/semigroup.{args.dim}.dat;
            rm ./data*.dat
        """],
        shell=True
    )


if __name__ == "__main__":
    main()
