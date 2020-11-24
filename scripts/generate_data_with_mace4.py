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
import sqlite3
import subprocess
from argparse import ArgumentParser, Namespace
from functools import partial
from multiprocessing.pool import Pool
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from neural_semigroups.utils import read_whole_file


def write_mace_input(
    indices: List[Tuple[int, int]], dim: int, filename: str
) -> None:
    """
    write a randomised file in a Mace4 format

    :param indices: a list of the known cells, as pairs (row, column)
    :param dim: total number of items in a magma
    :param filename: where to save the file
    :returns:
    """
    with open(filename, "w") as in_file:
        print(
            f"""
list(distinct).
[{", ".join(map(str, range(dim)))}].
end_of_list.
formulas(assumptions).
(x * y) * z = x * (y * z).
        """,
            file=in_file,
        )
        for i, j in indices:
            print(f"{i} * {j} = {np.random.randint(0, dim)}.", file=in_file)
        print("end_of_list.", file=in_file)


def table_completion(dim: int, task_id: int) -> Tuple[str, str]:
    """
    generate a random incomplete Cayley table and complete it

    :param dim: total number of items in a magma
    :param task_id: needed for using with multiprocessing
    :returns:
    """
    np.random.seed(task_id)
    dim_square = dim * dim
    known_cells_num = np.random.randint(1, dim_square)
    indices = [
        divmod(pair, dim)
        for pair in np.random.choice(
            range(dim_square), known_cells_num
        ).tolist()
    ]
    write_mace_input(indices, dim, f"{task_id}.in")
    subprocess.call(
        [f"mace4 -n {dim} < {task_id}.in > {task_id}.out 2> {task_id}.err"],
        shell=True,
    )
    output = read_whole_file(f"{task_id}.out")
    errors = read_whole_file(f"{task_id}.err")
    subprocess.call(
        [f"rm {task_id}.in {task_id}.out {task_id}.err"], shell=True
    )
    return (output, errors)


def parse_args() -> Namespace:
    """
    :returns: arguments namespace for the script
    """
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--dim", type=int, required=True)
    argument_parser.add_argument("--number_of_tasks", type=int, required=True)
    argument_parser.add_argument(
        "--number_of_processes", type=int, required=True
    )
    argument_parser.add_argument("--database_name", type=str, required=True)
    args = argument_parser.parse_args()
    return args


def main():
    """ do all """
    args = parse_args()
    with Pool(processes=args.number_of_processes) as pool, sqlite3.connect(
        args.database_name
    ) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'mace_output'"
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "CREATE TABLE mace_output(output STRING, errors STRING)"
            )
        with tqdm(total=args.number_of_tasks) as progress_bar:
            for output, errors in pool.imap_unordered(
                partial(table_completion, args.dim),
                range(args.number_of_tasks),
            ):
                cursor.execute(
                    "INSERT INTO mace_output VALUES(?, ?)", (output, errors)
                )
                progress_bar.update()
        cursor.close()


if __name__ == "__main__":
    main()
