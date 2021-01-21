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
import gzip
import shutil
import sqlite3
from os import listdir
from os.path import getmtime, join, splitext
from typing import List, Tuple

import requests
import torch
from torch import Tensor
from tqdm import tqdm

from neural_semigroups.constants import CURRENT_DEVICE
from neural_semigroups.magma import Magma, get_two_indices_per_sample

# the Cayley table of Klein Vierergruppe
# pylint: disable=not-callable
FOUR_GROUP = torch.tensor(
    [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
)

# some non associative magma
# (0 * 1) * 2 = 0 * 2 = 0
# 0 * (1 * 2) = 0 * 0 = 1
# pylint: disable=not-callable
NON_ASSOCIATIVE_MAGMA = torch.tensor([[1, 0, 0], [2, 2, 0], [2, 2, 2]])


def random_semigroup(
    dim: int, maximal_tries: int
) -> Tuple[bool, torch.Tensor]:
    """
    randomly search for a semigroup Cayley table.
    Not recommended to use with dim > 4

    :param dim: number of elements in a semigroup
    :param maximal_tries: how many times to try at most
    :returns: a pair (whether the Cayley table is associative, a Cayley table of a magma)

    """
    associative = False
    try_count = 0
    magma = Magma(cardinality=dim)
    while not associative and try_count <= maximal_tries:
        magma = Magma(cardinality=dim)
        associative = magma.is_associative
        try_count += 1
    return associative, magma.cayley_table


def get_magma_by_index(cardinality: int, index: int) -> Magma:
    """
    find a magma from a lexicographical order by its index

    :param cardinality: the number of elements in a magma
    :param index: an index of magma in a lexicographical order
    :returns: a magma with a given index
    """
    square = cardinality ** 2
    if index < 0 or index >= cardinality ** square:
        raise ValueError(
            """
        An index must be non negative and less than $n^(n^2)$"""
        )
    cayley_table = list()
    residual = index
    for _ in range(square):
        cayley_table.append(residual % cardinality)
        residual = residual // cardinality
    return Magma(
        torch.tensor(list(reversed(cayley_table))).reshape(
            cardinality, cardinality
        )
    )


def import_smallsemi_format(lines: List[str]) -> torch.Tensor:
    """
    imports lines in a format used by ``smallsemi`` `GAP package`.
    Format description:

    * filename is of a form ``data[n].gl``, :math:`1<=n<=7`
    * lines are separated by a pair of symbols ``\\r\\n``
    * there are exactly :math:`n^2` lines in a file
    * the first line is a header starting with '#' symbol
    * each line is a string of :math:`N` digits from :math:`0` to :math:`n-1`
    * :math:`N` is the number of semigroups in the database
    * each column represents a serialised Cayley table
    * the database contains only cells starting from the second
    * the first cell of each Cayley table is assumed to be filled with ``0``

    :param lines: lines read from a file of `smallsemi` format
    :returns: a list of Cayley tables

    .. _GAP package: https://www.gap-system.org/Manuals/pkg/smallsemi-0.6.12/doc/chap0.html

    """
    raw_tables = torch.tensor(
        [list(map(int, list(line[:-1]))) for line in lines[1:]]
    ).transpose(0, 1)
    tables = torch.cat(
        [torch.zeros([raw_tables.shape[0], 1], dtype=torch.long), raw_tables],
        dim=-1,
    )
    cardinality = int(tables.max()) + 1
    return tables.reshape(tables.shape[0], cardinality, cardinality)


def download_file_from_url(
    url: str, filename: str, buffer_size: int = 1024
) -> None:
    """
    downloads some file from the Web to a specified destination

    >>> import os
    >>> from neural_semigroups.constants import TEST_TEMP_DATA
    >>> temp_file = os.path.join(TEST_TEMP_DATA, "test.html")
    >>> if os.path.exists(temp_file):
    ...     os.remove(temp_file)
    >>> download_file_from_url("https://python.org/", temp_file)
    >>> os.path.exists(temp_file)
    True

    :param url: a valid HTTP URL
    :param filename: a valid filename
    :param buffer_size: a number of bytes to read from URL at once
    """
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    progress = tqdm(
        response.iter_content(chunk_size=buffer_size),
        f"Downloading {filename}",
        total=int(file_size / buffer_size),
        unit="kB",
    )
    with open(filename, "wb") as file:
        for data in progress:
            file.write(data)
    response.close()


def find_substring_by_pattern(
    strings: List[str], starts_with: str, ends_before: str
) -> str:
    """
    search for a first occurrence of a given pattern in a string list

    >>> some_strings = ["one", "two", "three"]
    >>> find_substring_by_pattern(some_strings, "t", "o")
    'tw'
    >>> find_substring_by_pattern(some_strings, "four", "five")
    Traceback (most recent call last):
       ...
    ValueError: pattern four.*five not found

    :param strings: a list of strings where the pattern is searched for
    :param starts_with: the first letters of a pattern
    :param ends_before: a substring which marks the beginning of something different
    :returns: a pattern which starts with ``starts_with`` and ends before ``ends_before``
    """
    for package_name in strings:
        starting_index = package_name.find(starts_with)
        if starting_index >= 0:
            ending_index = package_name.find(ends_before)
            return package_name[starting_index:ending_index]
    raise ValueError(f"pattern {starts_with}.*{ends_before} not found")


def get_newest_file(dir_path: str) -> str:
    """
    get the last modified file from a diretory

    >>> from pathlib import Path
    >>> from os import makedirs, path
    >>> from neural_semigroups.constants import TEST_TEMP_DATA
    >>> shutil.rmtree(path.join(TEST_TEMP_DATA, "tmp"), ignore_errors=True)
    >>> makedirs(path.join(TEST_TEMP_DATA, "tmp"))
    >>> Path(path.join(TEST_TEMP_DATA, "tmp", "one")).touch()
    >>> from time import sleep
    >>> sleep(0.01)
    >>> Path(path.join(TEST_TEMP_DATA, "tmp", "two")).touch()
    >>> get_newest_file(path.join(TEST_TEMP_DATA, "tmp"))
    'test_temp_data/tmp/two'

    :param dir_path: a directory path
    :returns: the last modified file's name
    """
    return max(
        [join(dir_path, filename) for filename in listdir(dir_path)],
        key=getmtime,
    )


def make_discrete(cayley_cubes: torch.Tensor) -> torch.Tensor:
    """
    transforms a batch of probabilistic Cayley cubes and in the following way:

    * maximal probabilities in the last dimension become ones
    * all other probabilies become zeros

    >>> make_discrete(torch.tensor([
    ...    [[[0.9, 0.1], [0.1, 0.9]], [[0.8, 0.2], [0.2, 0.8]]],
    ...    [[[0.7, 0.3], [0.3, 0.7]], [[0.7, 0.3], [0.3, 0.7]]],
    ... ]))
    tensor([[[[1., 0.],
              [0., 1.]],
    <BLANKLINE>
             [[1., 0.],
              [0., 1.]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[1., 0.],
              [0., 1.]],
    <BLANKLINE>
             [[1., 0.],
              [0., 1.]]]])

    :param cayley_cube: a batch of probabilistic cubes representing Cayley tables
    :returns: a batch of probabilistic cubes filled in with ``0`` or ``1``
    """
    cardinality = cayley_cubes.shape[1]
    batch_size = cayley_cubes.shape[0]
    cayley_table = cayley_cubes.max(dim=-1)[1]
    cube = torch.zeros(
        [batch_size, cardinality, cardinality, cardinality],
        dtype=torch.float32,
    )
    sample_index, one, two = get_two_indices_per_sample(
        batch_size, cardinality
    )
    cube[sample_index, one, two, cayley_table[sample_index, one, two]] = 1.0
    return cube


def count_different(one: Tensor, two: Tensor) -> Tensor:
    """
    given two batches of the same size, counts number of positions in these
    batches, on which the tensor from the first batch differs from the second

    :param one: one batch of tensors
    :param two: another batch of tensors
    :returns: the number of different tensors
    """
    batch_size = one.shape[0]
    return (
        torch.zeros(batch_size)
        .where(
            torch.abs(one - two).reshape(batch_size, -1).max(dim=1)[0] > 0,
            torch.ones(batch_size),
        )
        .sum()
    )


def hide_cells(cayley_table: Tensor, number_of_cells: int) -> Tensor:
    """
    set several cells in a Cayley table to math:`-1`

    >>> torch.manual_seed(42) # doctest: +ELLIPSIS
    <torch...
    >>> hide_cells(torch.tensor([[0, 1], [2, 3]]), 2).cpu()
    tensor([[ 0,  1],
            [-1, -1]])

    :param cayley_table: a Cayley table
    :param number_of_cells: a number of cells to hide
    :returns: a Cayley table with hidden cells
    """
    partial_table = cayley_table.clone()
    cardinality = cayley_table.shape[0]
    pairs = torch.randperm(cardinality * cardinality)[:number_of_cells]
    partial_table[pairs // cardinality, pairs % cardinality] = -1
    return partial_table


def read_whole_file(filename: str) -> str:
    """
    reads the whole file into a string,  for example

    >>> read_whole_file("README.md").split("\\n")[2]
    '# Neural Semigroups'

    :param filename: a name of the file to read
    :returns: whole contents of the file
    """
    with open(filename, "r") as input_file:
        text = input_file.read()
    return text


def partial_table_to_cube(table: Tensor) -> Tensor:
    """
    create a probabilistic cube from a partially filled Cayley table
    ``-1`` is translated to :math:`\\frac1n` where :math:`n` is table's
    cardinality, for example

    >>> partial_table_to_cube(torch.tensor([[0, -1], [0, 0]])).cpu()
    tensor([[[1.0000, 0.0000],
              [0.5000, 0.5000]],
    <BLANKLINE>
             [[1.0000, 0.0000],
              [1.0000, 0.0000]]])

    :param table: a Cayley table, partially filled by ``-1``'s
    :returns: a probabilistic cube
    """
    cardinality = table.shape[0]
    cube = torch.zeros(
        [cardinality, cardinality, cardinality],
        dtype=torch.float32,
        device=CURRENT_DEVICE,
    )
    rows, cols = torch.where(torch.ne(table, -1))
    cube[rows, cols, table[rows, cols]] = 1.0
    rows, cols = torch.where(torch.eq(table, -1))
    cube[rows, cols, :] = 1 / cardinality
    return cube.reshape([cardinality, cardinality, cardinality])


def create_table_if_not_exists(
    database_name: str, table_name, columns: List[str]
) -> None:
    """
    create a table if it does not exist

    :param database_name: where to create a table
    :param table_name: what table to create
    :param columns: a list of strings of format "COLUMN_NAME COLUMN_TYPE"
    :returns:
    """
    with sqlite3.connect(database_name, isolation_level=None) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        cursor = connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = ?", [table_name]
        )
        if cursor.fetchone()[0] == 0:
            connection.execute(
                f"CREATE TABLE {table_name}({', '.join(columns)})"
            )
        cursor.close()


def insert_values_into_table(
    database_name: str, table_name: str, values: Tuple[str]
) -> None:
    """
    inserts values into a table

    :param database_name:
    :param table_name:
    :param values:
    :returns:
    """
    with sqlite3.connect(database_name, isolation_level=None) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        cursor = connection.cursor()
        placeholders = ", ".join(len(values) * ["?"])
        cursor.execute(
            f"INSERT INTO {table_name} VALUES({placeholders})", values
        )
        cursor.close()


def gunzip(archive_path: str) -> None:
    """
    extracts a GZIP file in the same folder

    :param archive_path: a path ending with ``.gz``
    :returns:
    """
    with gzip.open(archive_path, "rb") as f_in:
        with open(splitext(archive_path)[0], "wb",) as f_out:
            shutil.copyfileobj(f_in, f_out)
