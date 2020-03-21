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
from itertools import permutations
from typing import List, Tuple

import numpy as np

from neural_semigroups.magma import Magma

# the Cayley table of Klein Vierergruppe
FOUR_GROUP = np.array([
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0]
])

# some non associative magma
# (0 * 1) * 2 = 0 * 2 = 0
# 0 * (1 * 2) = 0 * 0 = 1
NON_ASSOCIATIVE_MAGMA = np.array([
    [1, 0, 0], [2, 2, 0], [2, 2, 2]
])


def random_magma(cardinality: int) -> np.ndarray:
    """
    randomly generate a Cayley table for a magma

    :param cardinality: number of elements in a magma
    :returns: Cayley table of some magma
    """
    return np.random.randint(
        low=0,
        high=cardinality,
        size=cardinality * cardinality
    ).reshape(cardinality, cardinality)


def random_semigroup(dim: int, maximal_tries: int) -> Tuple[bool, np.ndarray]:
    """
    randomly serch for a semigroup Cayley table.
    Not recommended to use with dim > 4

    :param dim: number of elements in a semigroup
    :param maximal_tries: how many times to try at most
    :returns: a pair (whether the Cayley table is associative,
    a Cayley table of a magma)
    """
    associative = False
    try_count = 0
    while not associative and try_count <= maximal_tries:
        mult = random_magma(dim)
        associative = Magma(mult).is_associative
        try_count += 1
    return associative, mult


def next_magma(magma: Magma) -> Magma:
    """
    goes to the next magma Cayley table in their lexicographical
    order

    :param magma: a magma
    :returns: a magma
    """
    next_table = magma.cayley_table.copy()
    one = 1
    row = magma.cardinality - 1
    column = magma.cardinality - 1
    while one == 1:
        if next_table[row, column] < magma.cardinality - 1:
            next_table[row, column] += 1
            one = 0
        else:
            if column > 0:
                next_table[row, column] = 0
                column -= 1
            else:
                if row > 0:
                    next_table[row, column] = 0
                    row -= 1
                    column = magma.cardinality - 1
                else:
                    raise Exception("there is no next magma!")
    return Magma(next_table)


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
        filename_parts = filename.split(".")
        if len(filename_parts) != 3:
            wrong_name = True
        elif filename_parts[0] not in ("semigroup", "monoid", "group"):
            wrong_name = True
        elif filename_parts[2] != "npz":
            wrong_name = True
        elif not filename_parts[1].isdigit():
            wrong_name = True
        else:
            cardinality = int(filename_parts[1])
    if wrong_name:
        raise ValueError(
            f"""filename should be of format
[semigroup|monoid|group].[int].npz, not {filename}"""
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
        filename_parts = filename.split(".")
        if len(filename_parts) != 2:
            wrong_name = True
        elif filename_parts[1] != "gl":
            wrong_name = True
        elif filename_parts[0][:-1] != "data":
            wrong_name = True
        elif not filename_parts[0][-1].isdigit():
            wrong_name = True
        else:
            cardinality = int(filename_parts[0][-1])
    if wrong_name:
        raise ValueError(
            f"filename should be of format data[1-7].gl, not {filename}"
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
        raise ValueError("""
        An index must be non negative and less than $n^(n^2)$""")
    cayley_table = list()
    residual = index
    for _ in range(square):
        cayley_table.append(residual % cardinality)
        residual = residual // cardinality
    return Magma(np.array(
        list(reversed(cayley_table))
    ).reshape(cardinality, cardinality))


def import_smallsemi_format(lines: List[str]) -> np.ndarray:
    """
    imports lines in a format used by `smallsemi` GAP package
    https://www.gap-system.org/Manuals/pkg/smallsemi-0.6.11/doc/chap0.html

    :param lines: lines read from a file of `smallsemi` format
    :returns: a list of Cayley tables
    """
    raw_tables = np.array(
        [list(map(int, list(line[:-1]))) for line in lines[1:]]
    ).T
    tables = np.hstack([
        np.zeros([raw_tables.shape[0], 1], dtype=int),
        raw_tables
    ])
    cardinality = int(tables.max()) + 1
    return tables.reshape(tables.shape[0], cardinality, cardinality)


def get_isomorphic_magmas(cayley_table: np.ndarray) -> np.ndarray:
    """
    given a Cayley table of a magma generate Cayley tables of isomorphic magmas
    by appying all possible permutations of the magma's elements

    :param cayley_table: a Cayley table of a magma
    :returns: a list of Cayley tables of isomorphic magmas
    """
    isomorphic_cayley_tables = list()
    dim = cayley_table.shape[0]
    for permutation in permutations(range(dim)):
        isomorphic_cayley_table = np.array(np.zeros_like(
            cayley_table, dtype=int
        ))
        for i in range(dim):
            for j in range(dim):
                isomorphic_cayley_table[
                    permutation[i], permutation[j]
                ] = permutation[cayley_table[i, j]]
        isomorphic_cayley_tables.append(isomorphic_cayley_table)
    return np.unique(np.array(isomorphic_cayley_tables), axis=0)


def get_anti_isomorphic_magmas(cayley_table: np.ndarray) -> np.ndarray:
    """
    given a Cayley table of a magma generate Cayley tables of anti-isomorphic
    magmas by appying all possible permutations of the magma's elements

    :param cayley_table: a Cayley table of a magma
    :returns: a list of Cayley tables of anti-isomorphic magmas
    """
    anti_isomorphic_cayley_tables = list()
    dim = cayley_table.shape[0]
    for permutation in permutations(range(dim)):
        anti_isomorphic_cayley_table = np.array(np.zeros_like(
            cayley_table, dtype=int
        ))
        for i in range(dim):
            for j in range(dim):
                anti_isomorphic_cayley_table[
                    permutation[i], permutation[j]
                ] = permutation[cayley_table[j, i]]
        anti_isomorphic_cayley_tables.append(anti_isomorphic_cayley_table)
    return np.unique(np.array(anti_isomorphic_cayley_tables), axis=0)


def get_equivalent_magmas(cayley_table: np.ndarray) -> np.ndarray:
    """
    given a Cayley table of a magma generate Cayley tables of isomorphic and
    anti-isomorphic magmas

    :param cayley_table: a Cayley table of a magma
    :returns: a list of Cayley tables of isomorphic and anti-isomorphic magmas
    """
    equivalent_tables = np.concatenate([
        get_isomorphic_magmas(cayley_table),
        get_anti_isomorphic_magmas(cayley_table)
    ], axis=0)
    return np.unique(equivalent_tables, axis=0)
