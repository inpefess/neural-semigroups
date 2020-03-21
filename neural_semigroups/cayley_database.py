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
from os.path import basename
from typing import List, Tuple

import numpy as np
import torch
from torch.nn import Module
from tqdm import tqdm

from neural_semigroups.utils import (check_filename, check_smallsemi_filename,
                                     get_equivalent_magmas,
                                     import_smallsemi_format)


class CayleyDatabase:
    """
    a database of Cayley tables with different utility functions
    """
    # the number of elements in an underlying magma
    cardinality: int
    # a database of known Cayley tables of shape [:, cardinality, cardinality]
    database: np.ndarray
    # a pre-trained PyTorch Model
    model: Module
    # 1-D labels array of the same length as the database
    labels: np.ndarray

    def load_database(self, filename: str) -> None:
        """
        loads a known Cayley tables database from a file
        Database file description:

        * a filename is of a form [description].[n].npz
        * a file is of ``npz`` format readable by numpy

        :param filename: where to load a database from
        :returns:
        """
        self.cardinality = check_filename(basename(filename))
        npz_file = np.load(filename)
        self.database = npz_file["database"]
        self.labels = npz_file.get(
            "labels", np.zeros(len(self.database), dtype=np.int64)
        )
        npz_file.close()

    def load_smallsemi_database(self, filename: str) -> None:
        """
        loads a known Cayley tables database from a file in ``smallsemi`` format
        Format description:

        * filename is of a form ``data[n].gl``, :math:`1<=n<=7`
        * lines are separated by a pair of symbols ``\\r\\n``
        * there are exactly :math:`n^2` lines in a file
        * the first line is a header starting with '#' symbol
        * each line is a string of :math:`N` digits from :math:`0` to :math:`n-1`
        * :math`N` is the number of semigroups in the database
        * each column represents a serialised Cayley table
        * the database contains only cells starting from the second
        * the first cell of each Cayley table is assumed to be filled with ``0``

        :param filename: where to load a database from
        :returns:
        """
        self.cardinality = check_smallsemi_filename(basename(filename))
        with open(filename, "r") as database:
            self.database = import_smallsemi_format(database.readlines())
        self.labels = np.ones(len(self.database), dtype=np.int64)

    def augment_by_equivalent_tables(self) -> None:
        """
        for every Cayley table in a previously loaded database adds all of its
        equivalent tables to the database
        """
        database: List[np.ndarray] = []
        for table in tqdm(self.database):
            database += [get_equivalent_magmas(table)]
        self.database = np.unique(np.concatenate(database, axis=0), axis=0)

    def _check_input(self, cayley_table: List[List[int]]) -> bool:
        """
        checks the input to be a correct Cayley table

        :param cayley_table: a partially filled Cayley table (unknow entries are filled by ``-1``)
        :returns: whether the input is correct
        """
        correct = True
        table = np.array(cayley_table)
        if table.shape != (self.cardinality, self.cardinality):
            correct = False
        elif table.dtype != int:
            correct = False
        elif table.max() >= self.cardinality or table.min() < -1:
            correct = False
        return correct

    def search_database(
            self,
            cayley_table: List[List[int]]
    ) -> List[np.ndarray]:
        """
        get a list of possible completions of a partially filled Cayley table (unknow entries are filled by ``-1``)

        :param cayley_table: a partially filled Cayley table (unknow entries are filled by ``-1``)
        :returns: a list of Cayley tables
        """
        if not self._check_input(cayley_table):
            raise ValueError(
                f"invalid Cayley table of {self.cardinality} elements"
            )
        completions = list()
        if self.database is not None:
            partial_table = np.array(cayley_table)
            rows, cols = np.where(partial_table != -1)
            for table in tqdm(self.database):
                if np.alltrue(table[rows, cols] == partial_table[rows, cols]):
                    completions.append(table)
        return completions

    def fill_in_with_model(
            self,
            cayley_table: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        get a list of possible completions of a partially filled Cayley
        table (unknow entries are filled by ``-1``) using a machine learning model

        :param cayley_table: a partially filled Cayley table (unknow entries are filled by ``-1``)
        :returns: a tuple: (most probable completion, probabilistic cube)
        """
        if not self._check_input(cayley_table):
            raise ValueError(
                f"invalid Cayley table of {self.cardinality} elements"
            )
        table = np.array(cayley_table)
        inv_cardinality = 1 / self.cardinality
        cube = np.zeros(
            [self.cardinality, self.cardinality, self.cardinality],
            dtype=np.float32
        )
        rows, cols = np.where(table != -1)
        cube[rows, cols, table[rows, cols]] = 1.0
        rows, cols = np.where(table == -1)
        cube[rows, cols, :] = inv_cardinality
        prediction = self.model(torch.from_numpy(
            cube.reshape([
                -1, self.cardinality, self.cardinality, self.cardinality
            ])
        )).detach().numpy()[0]
        return (prediction.argmax(axis=-1), prediction)

    def load_model(self, filename: str) -> None:
        """
        load pre-trained PyTorch model

        :param filename: where to load the model from
        :returns:
        """
        self.model = torch.load(filename)

    def train_test_split(
            self,
            train_size: int,
            validation_size: int
    ) -> Tuple["CayleyDatabase", "CayleyDatabase", "CayleyDatabase"]:
        """
        split a database of Cayley table in three: train, validation, and test

        :param cayley_db: a database of Cayley tables
        :param train_size: number of tables in a train set
        :param train_size: number of tables in a validation set
        :returns: a triple of distinct Cayley tables databases: ``(train, validation, test)``
        """
        all_indices = np.arange(len(self.database))
        np.random.shuffle(all_indices)
        train_indices = all_indices[:train_size]
        validation_indices = all_indices[
            train_size:train_size + validation_size
        ]
        test_indices = all_indices[train_size + validation_size:]
        train = CayleyDatabase()
        train.database = self.database[train_indices]
        train.labels = self.labels[train_indices]
        validation = CayleyDatabase()
        validation.database = self.database[validation_indices]
        validation.labels = self.labels[validation_indices]
        test = CayleyDatabase()
        test.database = self.database[test_indices]
        test.labels = self.labels[test_indices]
        return train, validation, test
