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
import tarfile
from itertools import permutations
from os import listdir, makedirs, path, rename
from os.path import basename, getmtime, join
from shutil import rmtree
from typing import List, Tuple

import pandas as pd
import requests
import torch
from tqdm import tqdm

from neural_semigroups.constants import CURRENT_DEVICE, GAP_PACKAGES_URL
from neural_semigroups.magma import Magma

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
    randomly serch for a semigroup Cayley table.
    Not recommended to use with dim > 4

    :param dim: number of elements in a semigroup
    :param maximal_tries: how many times to try at most
    :returns: a pair (whether the Cayley table is associative, a Cayley table of a magma)

    """
    associative = False
    try_count = 0
    while not associative and try_count <= maximal_tries:
        mult = Magma(cardinality=dim)
        associative = mult.is_associative
        try_count += 1
    return associative, mult.cayley_table


def check_filename(filename: str) -> int:
    """
    checks filename, raises if it's incorrect

    :param filename: filename to check
    :returns: magma cardinality extracted from the filename
    """
    wrong_name = False
    if not isinstance(filename, str):
        wrong_name = True
    else:
        base_filename = basename(filename)
        filename_parts = base_filename.split(".")
        if len(filename_parts) != 3:
            wrong_name = True
        elif (
            filename_parts[0] not in ("semigroup", "monoid", "group")
            or filename_parts[2] != "zip"
            or not filename_parts[1].isdigit()
        ):
            wrong_name = True
        else:
            cardinality = int(filename_parts[1])
    if wrong_name:
        raise ValueError(
            "filename should be of format"
            f"[semigroup|monoid|group].[int].zip, not {base_filename}"
        )
    return cardinality


def check_smallsemi_filename(filename: str) -> int:
    """
    checks a filename from a `smallsemi` package, raises if it's incorrect

    :param filename: filename from a `smallsemi` package to check
    :returns: magma cardinality extracted from the filename
    """
    wrong_name = False
    if not isinstance(filename, str):
        wrong_name = True
    else:
        filename_parts = basename(filename).split(".")
        if len(filename_parts) != 3:
            wrong_name = True
        elif (
            filename_parts[2] != "gz"
            or filename_parts[1] != "gl"
            or filename_parts[0][:-1] != "data"
            or not filename_parts[0][-1].isdigit()
        ):
            wrong_name = True
        else:
            cardinality = int(filename_parts[0][-1])
            if cardinality < 2 or cardinality > 7:
                wrong_name = True
    if wrong_name:
        raise ValueError(
            "filename should be of format data[2-7].gl.gz" f" not {filename}"
        )
    return cardinality


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


def import_smallsemi_format(lines: List[bytes]) -> torch.Tensor:
    """
    imports lines in a format used by ``smallsemi`` `GAP package`.
    Format description:

    * filename is of a form ``data[n].gl.gz``, :math:`1<=n<=7`
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
        [list(map(int, list(line.decode("utf-8")[:-1]))) for line in lines[1:]]
    ).T
    tables = torch.cat(
        [torch.zeros([raw_tables.shape[0], 1], dtype=torch.long), raw_tables],
        dim=-1,
    )
    cardinality = int(tables.max()) + 1
    return tables.reshape(tables.shape[0], cardinality, cardinality)


def get_isomorphic_magmas(cayley_table: torch.Tensor) -> torch.Tensor:
    """
    given a Cayley table of a magma generate Cayley tables of isomorphic magmas
    by appying all possible permutations of the magma's elements

    :param cayley_table: a Cayley table of a magma
    :returns: a list of Cayley tables of isomorphic magmas
    """
    isomorphic_cayley_tables = list()
    dim = cayley_table.shape[0]
    for permutation in permutations(range(dim)):
        permutation_tensor = torch.tensor(permutation, device=CURRENT_DEVICE)
        isomorphic_cayley_table = torch.zeros(
            cayley_table.shape, dtype=torch.long, device=CURRENT_DEVICE
        )
        for i in range(dim):
            for j in range(dim):
                isomorphic_cayley_table[
                    permutation_tensor[i], permutation_tensor[j]
                ] = permutation_tensor[cayley_table[i, j]]
        isomorphic_cayley_tables.append(isomorphic_cayley_table)
    return torch.unique(torch.stack(isomorphic_cayley_tables), dim=0)


def get_anti_isomorphic_magmas(cayley_table: torch.Tensor) -> torch.Tensor:
    """
    given a Cayley table of a magma generate Cayley tables of anti-isomorphic
    magmas by appying all possible permutations of the magma's elements

    :param cayley_table: a Cayley table of a magma
    :returns: a list of Cayley tables of anti-isomorphic magmas
    """
    anti_isomorphic_cayley_tables = list()
    dim = cayley_table.shape[0]
    for permutation in permutations(range(dim)):
        anti_isomorphic_cayley_table = torch.zeros(
            cayley_table.shape, dtype=torch.long, device=CURRENT_DEVICE
        )
        permutation_tensor = torch.tensor(permutation, device=CURRENT_DEVICE)
        for i in range(dim):
            for j in range(dim):
                anti_isomorphic_cayley_table[
                    permutation_tensor[i], permutation_tensor[j]
                ] = permutation_tensor[cayley_table[j, i]]
        anti_isomorphic_cayley_tables.append(anti_isomorphic_cayley_table)
    return torch.unique(torch.stack(anti_isomorphic_cayley_tables), dim=0)


def get_equivalent_magmas(cayley_table: torch.Tensor) -> torch.Tensor:
    """
    given a Cayley table of a magma generate Cayley tables of isomorphic and
    anti-isomorphic magmas

    :param cayley_table: a Cayley table of a magma
    :returns: a list of Cayley tables of isomorphic and anti-isomorphic magmas
    """
    equivalent_tables = torch.cat(
        [
            get_isomorphic_magmas(cayley_table),
            get_anti_isomorphic_magmas(cayley_table),
        ],
        dim=0,
    )
    return torch.unique(equivalent_tables, dim=0)


def download_file_from_url(
    url: str, filename: str, buffer_size: int = 1024
) -> None:
    """
    downloads some file from the Web to a specified destination

    >>> download_file_from_url("https://python.org/", "/tmp/test.html")
    >>> import subprocess
    >>> subprocess.run("ls /tmp/test.html", shell=True).returncode
    0

    :param url: a valid HTTP URL
    :param filename: a valid filename
    :param buffer_size: a number of bytes to read from URL at once
    """
    response = requests.get(url, stream=True)
    if not response.ok:
        raise ValueError(f"Wrong response from URL: {response.status_code}")
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


def download_smallsemi_data(data_path: str) -> None:
    """
    downloads, unzips and moves ``smallsemi`` data

    :param data_path: data storage path
    :returns:
    """
    package_names = requests.get(GAP_PACKAGES_URL).text
    for package_name in package_names.split("\n"):
        starting_index = package_name.find("smallsemi")
        if starting_index >= 0:
            ending_index = package_name.find(".tar.bz2")
            smallsemi_with_version = package_name[starting_index:ending_index]
    url = f"{GAP_PACKAGES_URL}{smallsemi_with_version}.tar.bz2"
    temp_path = path.join(data_path, "tmp")
    rmtree(temp_path, ignore_errors=True)
    makedirs(temp_path, exist_ok=True)
    archive_path = path.join(temp_path, path.basename(url))
    download_file_from_url(url=url, filename=archive_path)
    with tarfile.open(archive_path) as archive:
        archive.extractall(temp_path)
    rename(
        path.join(temp_path, smallsemi_with_version, "data", "data2to7"),
        path.join(data_path, "smallsemi_data"),
    )


def print_report(totals: torch.Tensor) -> pd.DataFrame:
    """
    print report in a pretty format

    >>> totals = torch.tensor([[4, 4], [0, 1], [1, 2]])
    >>> print_report(totals)
           puzzles  solved  (%)  hidden cells  guessed  in %
    level
    1      4             0    0             4        1    25
    2      4             1   25             8        2    25

    :param totals: a table with three columns:

    * a column with total number of puzzles per level
    * a column with numbers of correctly solved puzzles
    * numbers of correctly guessed cells in all puzzles

    :returns: the report in a form of ``pandas.DataFrame``

    """
    levels = torch.arange(1, totals.shape[1] + 1)
    hidden_cells = totals[0] * levels
    return pd.DataFrame(
        {
            "level": levels,
            "puzzles": totals[0],
            "solved": totals[1],
            "(%)": totals[1] * 100 // totals[0],
            "hidden cells": hidden_cells,
            "guessed": totals[2],
            "in %": totals[2] * 100 // hidden_cells,
        }
    ).set_index("level")


def get_newest_file(dir_path: str) -> str:
    """
    get the last modified file from a diretory

    >>> from pathlib import Path
    >>> rmtree("/tmp/tmp/", ignore_errors=True)
    >>> makedirs("/tmp/tmp/")
    >>> Path("/tmp/tmp/one").touch()
    >>> from time import sleep
    >>> sleep(0.01)
    >>> Path("/tmp/tmp/two").touch()
    >>> get_newest_file("/tmp/tmp/")
    '/tmp/tmp/two'

    :param path: a diretory path
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

    :param cayley_cubes: a batch of probabilistic cubes representing Cayley tables
    :returns: a batch of probabilistic cubes filled in with ``0`` or ``1``
    """
    cardinality = cayley_cubes.shape[1]
    batch_size = cayley_cubes.shape[0]
    cayley_table = cayley_cubes.max(dim=-1)[1]
    cube = torch.zeros(
        [batch_size, cardinality, cardinality, cardinality],
        dtype=torch.float32,
    )
    a_range = torch.arange(cardinality)
    one = a_range.repeat_interleave(cardinality).repeat(batch_size)
    two = a_range.repeat(cardinality).repeat(batch_size)
    sample_index = torch.arange(batch_size).repeat_interleave(
        cardinality * cardinality
    )
    cube[sample_index, one, two, cayley_table[sample_index, one, two]] = 1.0
    return cube
