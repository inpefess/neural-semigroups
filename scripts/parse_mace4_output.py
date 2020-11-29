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
import re
import sqlite3
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import torch
from tqdm import tqdm


def parse_args() -> Namespace:
    """
    :returns: arguments namespace for the script
    """
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--database_name", type=str, required=True)
    argument_parser.add_argument("--dataset_name", type=str, required=True)
    args = argument_parser.parse_args()
    return args


def get_cube_from_output(output: str, dim: int) -> torch.Tensor:
    """
    represents a partial magma from the output of ``mace4`` as a probabilistic cube

    :param output: output of ``mace4``
    :param dim: dimension of a magma
    :returns: a probabilistic representation of a partial magma
    """
    search_result = re.search(r".*INPUT(.*)end of input.*", output, re.DOTALL)
    if search_result is None:
        raise ValueError("wrong mace4 output file format!")
    input_lines = search_result.groups()[0]
    cube = torch.ones([dim, dim, dim]) / dim
    for equation in re.finditer(r"(\d+) \* (\d+) = (\d+)\.", input_lines):
        i, j, k = map(int, equation.groups())
        cube[i, j, :] = 0
        cube[i, j, k] = 1
    return cube


def connect_to_db(database_name: str) -> sqlite3.Cursor:
    """ open a connection to an SQLite database

    :param database_name: filename of a database
    :returns: a cursor to the database
    """
    connection = sqlite3.connect(database_name, isolation_level=None)
    connection.execute("PRAGMA journal_mode=WAL;")
    return connection.cursor()


def get_additional_info(cursor: sqlite3.Cursor) -> Tuple[int, int]:
    """
    gets some info from an SQLite database with ``mace4`` outputs

    :param cursor: an SQLite database cursor
    :returns: a total number of rows in a table, a magma dimension
    """
    cursor.execute("SELECT COUNT(*) FROM mace_output")
    row_count = cursor.fetchone()[0]
    cursor.execute("SELECT output FROM mace_output")
    search_result = re.search(r".* DOMAIN SIZE (\d+) .*", cursor.fetchone()[0])
    if search_result is not None:
        dim = int(search_result.groups()[0])
    else:
        raise ValueError("database is empty?")
    return row_count, dim


def final_steps(
    cursor: sqlite3.Cursor, dataset: dict, dataset_name: str
) -> None:
    """
    closing a database connection and writing the results to disk

    :param cursor: an SQLite cursor to close
    :param dataset: a dataset to save with ``torch``
    :param dataset_name: a filename for saving a dataset
    :returns:
    """
    cursor.close()
    cursor.connection.close()
    torch.save(dataset, dataset_name)


def get_features_and_labels(
    cursor: sqlite3.Cursor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    get data for a classification task from a table of ``mace4`` outputs

    :param cursor: an SQLite cursor
    :returns: features and labels for a binary classification task
    """
    row_count, dim = get_additional_info(cursor)
    cursor.execute("SELECT output FROM mace_output")
    features = list()
    labels = torch.zeros([row_count], dtype=torch.long)
    for row_index in tqdm(range(row_count)):
        output = cursor.fetchone()[0]
        features.append(get_cube_from_output(output, dim))
        if re.search(r".*Exiting with 1 model\..*", output) is not None:
            labels[row_index] = 1
    return features, labels


def main():
    """ do all """
    args = parse_args()
    cursor = connect_to_db(args.database_name)
    features, labels = get_features_and_labels(cursor)
    final_steps(
        cursor,
        {"features": torch.stack(features), "labels": labels},
        args.dataset_name,
    )


if __name__ == "__main__":
    main()
