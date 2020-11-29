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
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm

from neural_semigroups.constants import CAYLEY_DATABASE_PATH
from neural_semigroups.magma import Magma
from neural_semigroups.utils import (
    get_equivalent_magmas,
    load_data_and_labels_from_file,
    load_data_and_labels_from_smallsemi,
    partial_table_to_cube,
)


class CayleyDatabase:
    """
    a database of Cayley tables with different utility functions
    """

    _model: Optional[Module] = None

    def __init__(
        self,
        cardinality: int,
        database_filename: Optional[str] = None,
        data_path: str = CAYLEY_DATABASE_PATH,
    ):
        """
        :param cardinality: the number of elements in underlying magmas
        :param database_filename: a full path to a pre-generated Cayley database.
                                  If ``None``, a ``smallsemi`` data is used.
        :param data_path: a valid path to use as a permanent data storage
        """
        self.cardinality = cardinality
        self.data_path = data_path
        self.database, self.labels = (
            load_data_and_labels_from_smallsemi(cardinality, data_path)
            if database_filename is None
            else load_data_and_labels_from_file(database_filename)
        )

    def augment_by_equivalent_tables(self) -> None:
        """
        for every Cayley table in a previously loaded database adds all of its
        equivalent tables to the database
        """
        self.database = get_equivalent_magmas(self.database)

    def _check_input(self, cayley_table: List[List[int]]) -> bool:
        """
        checks the input to be a correct Cayley table

        :param cayley_table: a partially filled Cayley table (unknow entries are filled by ``-1``)
        :returns: whether the input is correct
        """
        if isinstance(cayley_table, list):
            # pylint: disable=not-callable
            table = torch.tensor(cayley_table)
            correct = (
                table.shape == torch.Size([self.cardinality, self.cardinality])
                and table.dtype == torch.long
                and table.max() < self.cardinality
                and table.min() >= -1
            )
        else:
            correct = False
        return correct

    def search_database(self, cayley_table: List[List[int]]) -> List[Tensor]:
        """
        get a list of possible completions of a partially filled Cayley table
        (unknown entries are filled by ``-1``)

        :param cayley_table: a partially filled Cayley table (unknow entries are filled by ``-1``)
        :returns: a list of Cayley tables
        """
        if not self._check_input(cayley_table):
            raise ValueError(
                f"invalid Cayley table of {self.cardinality} elements"
            )
        completions = list()
        if self.database is not None:
            # pylint: disable=not-callable
            partial_table = torch.tensor(cayley_table)
            rows, cols = torch.where(partial_table != -1)
            for table in tqdm(
                self.database, desc="full scan over Cayley database"
            ):
                if torch.allclose(
                    table[rows, cols], partial_table[rows, cols]
                ):
                    completions.append(table)
        return completions

    def fill_in_with_model(
        self, cayley_table: List[List[int]]
    ) -> Tuple[Tensor, Tensor]:
        """
        get a list of possible completions of a partially filled Cayley table
        (unknow entries are filled by ``-1``) using a machine learning model

        :param cayley_table: a partially filled Cayley table (unknow entries are filled by ``-1``)
        :returns: a tuple: (most probable completion, probabilistic cube)
        """
        self.model.eval()
        if not self._check_input(cayley_table):
            raise ValueError(
                f"invalid Cayley table of {self.cardinality} elements"
            )
        # pylint: disable=not-callable
        cube = partial_table_to_cube(torch.tensor(cayley_table))
        prediction = self.model(cube).detach()[0]
        return (prediction.argmax(axis=-1), prediction)

    def load_model(self, filename: str) -> None:
        """
        load pre-trained PyTorch model

        :param filename: where to load the model from
        :returns:
        """
        self._model = torch.load(filename)

    def train_test_split(
        self, train_size: int, validation_size: int
    ) -> Tuple["CayleyDatabase", "CayleyDatabase", "CayleyDatabase"]:
        """
        split a database of Cayley table in three: train, validation, and test

        :param cayley_db: a database of Cayley tables
        :param train_size: number of tables in a train set
        :param validation_size: number of tables in a validation set
        :returns: a triple of distinct Cayley tables databases: ``(train, validation, test)``
        """
        splits = random_split(
            TensorDataset(self.database, self.labels),
            [
                train_size,
                validation_size,
                self.database.shape[0] - train_size - validation_size,
            ],
            generator=torch.default_generator,
        )
        train = CayleyDatabase(self.cardinality)
        train.database, train.labels = [
            torch.stack(i) for i in zip(*splits[0])  # type: ignore
        ]
        validation = CayleyDatabase(self.cardinality)
        validation.database, validation.labels = [
            torch.stack(i) for i in zip(*splits[1])  # type: ignore
        ]
        test = CayleyDatabase(self.cardinality)
        test.database, test.labels = [
            torch.stack(i) for i in zip(*splits[2])  # type: ignore
        ]
        return train, validation, test

    @property
    def model(self) -> Module:
        """
        :returns: pre-trained Torch model
        """
        if self._model is None:
            raise ValueError("The model should be loaded first!")
        return self._model

    @model.setter
    def model(self, model: Module) -> None:
        """
        :param model: pre-trained Torch model
        """
        self._model = model

    def testing_report(self, max_level: int = -1) -> Tensor:
        """
        this function:

        * takes 1000 random Cayley tables from the database
          (if there are less tables, it simply takes all of them)
        * for each Cayley table generates ``max_level`` puzzles
        * each puzzle is created from a table by omitting several cell values
        * for each table the function omits 1, 2, and up to ``max_level`` of all cells
        * each puzzle is given to a pre-trained model of that database
        * if the model returns an associative table
          (not necessary the original one)
          it is considered to be a sucessfull solution

        :param max_level: up to how many cells to omit when creating a puzzle;
            when not provided or explicitly set to ``-1`` it defaults to the total number of cells in a table
        :returns: statistics of solved puzzles splitted by the levels of difficulty
                  (number of cells omitted)
        """
        max_level = self.cardinality ** 2 if max_level == -1 else max_level
        totals = torch.zeros((2, max_level))
        test_indices = torch.randperm(len(self.database))[
            : min(len(self.database), 1000)
        ]
        for i in tqdm(test_indices, desc="generating and solving puzzles"):
            for level in range(1, max_level + 1):
                rows, cols = zip(
                    *[
                        (point // self.cardinality, point % self.cardinality)
                        for point in torch.randint(
                            0, self.cardinality ** 2, [level]
                        )
                    ]
                )
                puzzle = self.database[i].clone().detach()
                puzzle[rows, cols] = -1
                solution, _ = self.fill_in_with_model(puzzle.tolist())
                totals[0, level - 1] += 1
                totals[1, level - 1] += Magma(solution).is_associative
        return totals
