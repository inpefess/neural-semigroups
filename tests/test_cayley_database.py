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
from shutil import rmtree

# pylint: disable-all
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
from torch.nn import Module

from neural_semigroups.cayley_database import CayleyDatabase
from neural_semigroups.constants import CURRENT_DEVICE


class TestCayleyDatabase(TestCase):
    def setUp(self):
        torch.manual_seed(43)
        self.identity_module = Module()
        self.identity_module.forward = lambda x: x
        temp_data_path = "./test_temp_data"
        rmtree(temp_data_path, ignore_errors=True)
        self.cayley_db = CayleyDatabase(2, data_path=temp_data_path)
        self.cayley_db.database = torch.tensor(
            [
                [[0, 1], [1, 0]],
                [[1, 0], [1, 1]],
                [[1, 1], [1, 0]],
                [[1, 1], [1, 1]],
            ]
        )
        self.cayley_db.labels = torch.arange(4)

    @patch("torch.load")
    def test_load_database(self, torch_load_mock):
        database = torch.tensor([[[0, 1], [2, 3]], [[1, 0], [3, 2]]])
        npz_file = MagicMock()
        npz_file.__getitem__ = (
            lambda x, y: database if y == "database" else None
        )
        npz_file.get = lambda x, y: y if x == "labels" else None
        torch_load_mock.return_value = npz_file
        cayley_db = CayleyDatabase(2, "semigroup.2.zip")
        self.assertEqual(cayley_db.cardinality, 2)
        self.assertTrue(torch.allclose(cayley_db.database, database))
        self.assertTrue(
            torch.allclose(cayley_db.labels, torch.zeros(2, dtype=torch.long))
        )

    def test_search_database(self):
        self.cayley_db.cardinality = 2
        complete = self.cayley_db.search_database([[-1, 1], [1, 0]])
        self.assertIsInstance(complete, list)
        self.assertEqual(len(complete), 2)
        self.assertTrue(
            torch.allclose(complete[0], self.cayley_db.database[0])
        )
        self.assertTrue(
            torch.allclose(complete[1], self.cayley_db.database[2])
        )
        complete = self.cayley_db.search_database([[-1, -1], [0, 0]])
        self.assertIsInstance(complete, list)
        self.assertEqual(len(complete), 0)
        with self.assertRaises(ValueError):
            self.cayley_db.search_database("no good")

    def test_fill_in_with_model(self):
        self.cayley_db.cardinality = 2
        some_input = [[-1, 0], [0, 1]]
        self.cayley_db.model = self.identity_module
        table, cube = self.cayley_db.fill_in_with_model(some_input)
        self.assertIsInstance(table, torch.Tensor)
        self.assertIsInstance(cube, torch.Tensor)
        self.assertEqual(table.dtype, torch.int64)
        self.assertEqual(cube.dtype, torch.float)
        self.assertTrue(
            torch.allclose(
                table, torch.tensor([[0, 0], [0, 1]]).to(CURRENT_DEVICE)
            )
        )
        self.assertTrue(
            torch.allclose(
                cube,
                torch.tensor(
                    [[[0.5, 0.5], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]
                ).to(CURRENT_DEVICE),
            )
        )
        with self.assertRaises(ValueError):
            self.cayley_db.fill_in_with_model("no good")

    def test_check_input(self):
        self.cayley_db.cardinality = 2
        self.assertFalse(self.cayley_db._check_input([[0]]))
        self.assertFalse(self.cayley_db._check_input([[0.5, 0], [0, 0]]))
        self.assertFalse(self.cayley_db._check_input([[-2, 0], [0, 0]]))
        self.assertFalse(self.cayley_db._check_input([[2, 0], [0, 0]]))

    def test_augment_by_equivalent_tables(self):
        database = torch.tensor(
            [[[0, 1], [1, 0]], [[1, 0], [0, 1]], [[1, 0], [1, 1]]]
        ).to(CURRENT_DEVICE)
        true_database = torch.tensor(
            [
                [[0, 0], [1, 0]],
                [[0, 1], [0, 0]],
                [[0, 1], [1, 0]],
                [[1, 0], [0, 1]],
                [[1, 0], [1, 1]],
                [[1, 1], [0, 1]],
            ]
        ).to(CURRENT_DEVICE)
        self.cayley_db.database = database
        self.cayley_db.augment_by_equivalent_tables()
        self.assertTrue(torch.allclose(true_database, self.cayley_db.database))

    def test_train_test_split(self):
        train, validation, test = self.cayley_db.train_test_split(2, 1)
        self.assertTrue(
            train.database.allclose(self.cayley_db.database[[0, 1]])
        )
        self.assertTrue(train.labels.allclose(torch.tensor([0, 1])))
        self.assertTrue(
            validation.database.allclose(self.cayley_db.database[3])
        )
        self.assertTrue(validation.labels.allclose(torch.tensor([3])))
        self.assertTrue(test.database.allclose(self.cayley_db.database[2]))
        self.assertTrue(test.labels.allclose(torch.tensor([2])))

    @patch("torch.load")
    def test_load_model(self, load_mock):
        self.cayley_db.load_model("model")
        load_mock.assert_called_once()

    def test_model(self):
        with self.assertRaises(ValueError):
            self.cayley_db.model

    def test_testing_report(self):
        self.cayley_db.model = self.identity_module
        torch.manual_seed(777)
        self.assertTrue(
            self.cayley_db.testing_report(2).equal(
                torch.tensor([[4, 4], [2, 0]], dtype=torch.float)
            )
        )
