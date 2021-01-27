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
import re
import sqlite3
from typing import Callable, Optional

import torch
from tqdm import tqdm

from neural_semigroups.semigroups_dataset import SemigroupsDataset
from neural_semigroups.utils import connect_to_db


class Mace4Semigroups(SemigroupsDataset):
    """
    a ``torch.util.data.Dataset`` wrapper for the data of ``mace4`` output
    stored in a ``sqlite`` database

    >>> import shutil
    >>> from neural_semigroups.constants import TEST_TEMP_DATA
    >>> import os
    >>> from neural_semigroups.generate_data_with_mace4 import (
    ... generate_data_with_mace4)
    >>> shutil.rmtree(TEST_TEMP_DATA, ignore_errors=True)
    >>> os.mkdir(TEST_TEMP_DATA)
    >>> database = os.path.join(TEST_TEMP_DATA,"test.db")
    >>> torch.manual_seed(42) # doctest: +ELLIPSIS
    <torch...
    >>> generate_data_with_mace4([
    ... "--max_dim", "2",
    ... "--min_dim", "2",
    ... "--number_of_tasks", "1",
    ... "--database_name", database])
    >>> mace4_semigroups = Mace4Semigroups(
    ...     root=database,
    ...     cardinality=2,
    ...     transform=lambda x: x
    ... )
    >>> mace4_semigroups[0][0]
    tensor([[0, 0],
            [0, 0]])
    """

    _where_clause = "WHERE output LIKE '%Process % exit (max_models)%'"

    def __init__(
        self,
        cardinality: int,
        root: str,
        transform: Optional[Callable] = None,
    ):
        """
        :param root: a full path to an ``sqlite`` database file
           which has a table ``mace_output`` with a string column ``output``
        :param cardinality: the cardinality of semigroups
        :param transform: a function/transform that takes a Cayley table
            and returns a transformed version.
        """
        super().__init__(root, cardinality, transform)
        self.load_data_from_mace_output()

    def get_table_from_output(self, output: str) -> torch.Tensor:
        """
        gets a Cayley table of a magma from the output of ``mace4``

        :param output: output of ``mace4``
        :returns: a Cayley table
        """
        search_result = re.search(
            r".*function\(\*\(\_,\_\),\ \[(.*)\]\)\..*", output, re.DOTALL
        )
        if search_result is None:
            raise ValueError("wrong mace4 output file format!")
        input_lines = search_result.groups()[0]
        # pylint: disable=not-callable
        cayley_table = torch.tensor(
            list(
                map(
                    int,
                    input_lines.translate(
                        str.maketrans("", "", " \t\n])")
                    ).split(","),
                )
            )
        ).view(self.cardinality, self.cardinality)
        return cayley_table

    def get_additional_info(self, cursor: sqlite3.Cursor) -> int:
        """
        gets some info from an SQLite database with ``mace4`` outputs

        :param cursor: an SQLite database cursor
        :returns: a total number of rows in a table, a magma dimension
        """
        cursor.execute(
            f"SELECT COUNT(*) FROM mace_output {self._where_clause}"
        )
        row_count = cursor.fetchone()[0]
        return row_count

    def load_data_from_mace_output(self) -> None:
        """ loads data generated by ``mace4`` from an ``sqlite`` database """
        cursor = connect_to_db(self.root)
        row_count = self.get_additional_info(cursor)
        cursor.execute(f"SELECT output FROM mace_output {self._where_clause}")
        features = list()
        for _ in tqdm(range(row_count)):
            output = cursor.fetchone()[0]
            features.append(self.get_table_from_output(output))
        self.tensors = (torch.stack(features),)
