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
import os
import sqlite3
import subprocess
from argparse import ArgumentParser, Namespace
from functools import partial
from multiprocessing.pool import Pool
from typing import List, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from neural_semigroups.utils import read_whole_file


def generate_partial_table(cardinality: int, known_cells_num: int) -> Tensor:
    """
    generate a Cayley table in which some cells are set to :math:`-1` (unknown)

    >>> torch.eq(generate_partial_table(2, 3), -1).sum().item()
    1

    :param cardinality: magma cardinality
    :param known_cells_num: the number of cells to fill with numbers from :math:`0` to :math:`n-1`
    :returns: a square table with numbers from :math:`-1` to :math:`n-1`
    """
    table = -torch.ones((cardinality, cardinality))
    for pair in torch.randperm(cardinality * cardinality)[:known_cells_num]:
        row, col = divmod(pair.item(), cardinality)
        table[row, col] = torch.randint(cardinality, (1,)).item()
    return table


def write_mace_input(partial_table: Tensor, dim: int, filename: str) -> None:
    """
    write a randomised file in a Mace4 format

    :param partial_table: a Cayley table partially filled with :math:`-1`'s
    :param dim: total number of items in a magma
    :param filename: where to save the file
    :returns:
    """
    with open(filename, "w") as in_file:
        print("formulas(assumptions).", file=in_file)
        print("(x * y) * z = x * (y * z).", file=in_file)
        for i in range(dim):
            for j in range(dim):
                cell = int(partial_table[i, j].item())
                if cell != -1:
                    print(f"{i} * {j} = {cell}.", file=in_file)
        print("end_of_list.", file=in_file)


def table_completion(
    dims: List[int], mace_timeout: int, mace_memory_mb: int, task_id: int
) -> Tuple[str, str]:
    """
    generate a random incomplete Cayley table and complete it

    :param dims: cardinalities of magmas to be generated in each task
    :param mace_timeout: number of seconds for ``mace4`` to search for a model or :math:``-1`` if it can search forever
    :param mace_memory_mb: number of memory used by a single ``mace4`` process in megabytes
    :param task_id: needed for using with multiprocessing
    :returns:
    """
    dim = dims[task_id]
    partial_table = generate_partial_table(
        dim, int(torch.randint(1, dim * dim, (1,)).item())
    )
    write_mace_input(partial_table, dim, f"{task_id}.in")
    with open(f"{task_id}.in", "r") as task_in, open(
        f"{task_id}.out", "w"
    ) as task_out, open(f"{task_id}.err", "w") as task_err:
        subprocess.run(
            [
                "mace4",
                f"-n {dim}",
                f"-t {mace_timeout}",
                f"-b {mace_memory_mb}",
            ],
            stdin=task_in,
            stdout=task_out,
            stderr=task_err,
            check=False,
        )
    output = read_whole_file(f"{task_id}.out")
    errors = read_whole_file(f"{task_id}.err")
    for extension in ["in", "out", "err"]:
        os.remove(f"{task_id}.{extension}")
    return output, errors


def parse_args() -> Namespace:
    """
    :returns: arguments namespace for the script
    """
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--max_dim", type=int, required=True)
    argument_parser.add_argument("--min_dim", type=int, required=True)
    argument_parser.add_argument("--number_of_tasks", type=int, required=True)
    argument_parser.add_argument(
        "--number_of_processes", type=int, required=True
    )
    argument_parser.add_argument(
        "--mace_timeout", type=int, required=False, default=-1
    )
    argument_parser.add_argument(
        "--mace_memory_mb", type=int, required=False, default=500
    )
    argument_parser.add_argument("--database_name", type=str, required=True)
    args = argument_parser.parse_args()
    return args


def create_if_not_exist(database_name: str) -> None:
    """
    create a table ``mace_output`` if it does not exist

    :param database_name: where to create a table
    :returns:
    """
    with sqlite3.connect(database_name, isolation_level=None) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        cursor = connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'mace_output'"
        )
        if cursor.fetchone()[0] == 0:
            connection.execute(
                "CREATE TABLE mace_output(output STRING, errors STRING)"
            )
        cursor.close()


def write_mace_output(database_name: str, values: Tuple[str, str]) -> None:
    """
    inserts values into a table ``mace_output``

    :param database_name: where to create a table
    :param values: what to insert
    :returns:
    """
    with sqlite3.connect(database_name, isolation_level=None) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        cursor = connection.cursor()
        cursor.execute("INSERT INTO mace_output VALUES(?, ?)", values)
        cursor.close()


def main():
    """ do all """
    args = parse_args()
    dims = (
        torch.randint(args.min_dim, args.max_dim + 1, (args.number_of_tasks,))
        .numpy()
        .tolist()
    )
    with Pool(processes=args.number_of_processes) as pool:
        create_if_not_exist(args.database_name)
        with tqdm(total=args.number_of_tasks) as progress_bar:
            for output, errors in pool.imap_unordered(
                partial(
                    table_completion,
                    dims,
                    args.mace_timeout,
                    args.mace_memory_mb,
                ),
                range(args.number_of_tasks),
            ):
                write_mace_output(args.database_name, (output, errors))
                progress_bar.update()


if __name__ == "__main__":
    main()
