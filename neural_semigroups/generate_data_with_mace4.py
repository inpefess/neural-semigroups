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
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from neural_semigroups import Magma
from neural_semigroups.utils import (
    connect_to_db,
    create_table_if_not_exists,
    hide_cells,
    insert_values_into_table,
    read_whole_file,
)

TABLE_NAME = "mace_output"


def write_mace_input(partial_table: Tensor, dim: int, filename: str) -> None:
    """
    write a randomised file in a Mace4 format

    :param partial_table: a Cayley table partially filled with :math:`-1`'s
    :param dim: total number of items in a magma
    :param filename: where to save the file
    :returns:
    """
    with open(filename, "w", encoding="utf-8") as in_file:
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
    partial_table = hide_cells(
        Magma(cardinality=dim).cayley_table,
        int(torch.randint(1, dim * dim, (1,)).item()),
    )
    write_mace_input(partial_table, dim, f"{task_id}.in")
    with open(f"{task_id}.in", "r", encoding="utf-8") as task_in, open(
        f"{task_id}.out", "w", encoding="utf-8"
    ) as task_out, open(f"{task_id}.err", "w", encoding="utf-8") as task_err:
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


def parse_args(args: Optional[List[str]]) -> Namespace:
    """

    :param args: a list of string arguments
        (for testing and use in a non script scenario)
    :returns: arguments namespace for the script
    """
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--max_dim", type=int, required=True)
    argument_parser.add_argument("--min_dim", type=int, required=True)
    argument_parser.add_argument("--number_of_tasks", type=int, required=True)
    argument_parser.add_argument(
        "--number_of_processes", type=int, required=False, default=1
    )
    argument_parser.add_argument(
        "--mace_timeout", type=int, required=False, default=-1
    )
    argument_parser.add_argument(
        "--mace_memory_mb", type=int, required=False, default=500
    )
    argument_parser.add_argument("--database_name", type=str, required=True)
    parsed_args = argument_parser.parse_args(args)
    return parsed_args


def work_with_database(
    cursor: sqlite3.Cursor, args: Namespace, pool: Pool, dims: int
) -> None:
    """
    a function-helper

    :param cursor: a cursor for a database to work with
    :param args: additional arguments
    :param pool: a multiprocessing pool
    :param dims: semigroups cardinality
    :returns:
    """
    try:
        create_table_if_not_exists(
            cursor,
            TABLE_NAME,
            ["output STRING", "errors STRING"],
        )
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
                insert_values_into_table(cursor, TABLE_NAME, (output, errors))
                progress_bar.update()
    finally:
        pool.close()
        pool.join()


def generate_data_with_mace4(input_args: Optional[List[str]] = None) -> None:
    """
    :param input_args: a list of arguments
        (if ``None`` then ones from the command line are used)
    :returns:
    """
    args = parse_args(input_args)
    dims = (
        torch.randint(args.min_dim, args.max_dim + 1, (args.number_of_tasks,))
        .numpy()
        .tolist()
    )
    cursor = connect_to_db(args.database_name)
    pool = Pool(processes=args.number_of_processes)
    work_with_database(cursor, args, pool, dims)
    cursor.connection.close()
