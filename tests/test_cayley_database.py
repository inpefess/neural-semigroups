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
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

from neural_semigroups.cayley_database import CayleyDatabase


class TestCayleyDatabase(TestCase):
    def setUp(self):
        np.random.seed(43)
        self.cayley_db = CayleyDatabase(2, data_path="./tests")
        self.cayley_db.database = np.array(
            [
                [[0, 1], [1, 0]],
                [[1, 0], [1, 1]],
                [[1, 1], [1, 0]],
                [[1, 1], [1, 1]],
            ]
        )
        self.cayley_db.labels = np.arange(4)

    @patch("numpy.load")
    def test_load_database(self, numpy_load_mock):
        database = np.array([[[0, 1], [2, 3]], [[1, 0], [3, 2]]])
        npz_file = MagicMock()
        npz_file.__getitem__ = (
            lambda x, y: database if y == "database" else None
        )
        npz_file.get = lambda x, y: y if x == "labels" else None
        numpy_load_mock.return_value = npz_file
        with patch.object(npz_file, "close") as mock:
            cayley_db = CayleyDatabase(2, "semigroup.2.npz")
            mock.assert_called_once()
        self.assertEqual(cayley_db.cardinality, 2)
        self.assertTrue(np.allclose(cayley_db.database, database))
        self.assertTrue(np.allclose(cayley_db.labels, np.zeros(2)))

    def test_search_database(self):
        self.cayley_db.cardinality = 2
        complete = self.cayley_db.search_database([[-1, 1], [1, 0]])
        self.assertIsInstance(complete, list)
        self.assertEqual(len(complete), 2)
        self.assertTrue(np.allclose(complete[0], self.cayley_db.database[0]))
        self.assertTrue(np.allclose(complete[1], self.cayley_db.database[2]))
        complete = self.cayley_db.search_database([[-1, -1], [0, 0]])
        self.assertIsInstance(complete, list)
        self.assertEqual(len(complete), 0)
        with self.assertRaises(Exception):
            self.cayley_db.search_database("no good")

    def test_fill_in_with_model(self):
        self.cayley_db.cardinality = 2
        input = [[-1, 0], [0, 1]]
        self.cayley_db.model = lambda x: x
        table, cube = self.cayley_db.fill_in_with_model(input)
        self.assertIsInstance(table, np.ndarray)
        self.assertIsInstance(cube, np.ndarray)
        self.assertEqual(table.dtype, int)
        self.assertEqual(cube.dtype, np.float32)
        self.assertTrue(np.allclose(table, np.array([[0, 0], [0, 1]])))
        self.assertTrue(
            np.allclose(
                cube,
                np.array([[[0.5, 0.5], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]),
            )
        )
        with self.assertRaises(Exception):
            self.cayley_db.fill_in_with_model("no good")

    def test_check_input(self):
        self.cayley_db.cardinality = 2
        self.assertFalse(self.cayley_db._check_input([[0]]))
        self.assertFalse(self.cayley_db._check_input([[0.5, 0], [0, 0]]))
        self.assertFalse(self.cayley_db._check_input([[-2, 0], [0, 0]]))
        self.assertFalse(self.cayley_db._check_input([[2, 0], [0, 0]]))

    def test_augment_by_equivalent_tables(self):
        database = np.array(
            [
                np.array([[0, 1], [1, 0]]),
                np.array([[1, 0], [0, 1]]),
                np.array([[1, 0], [1, 1]]),
            ]
        )
        true_database = np.array(
            [
                np.array([[0, 0], [1, 0]]),
                np.array([[0, 1], [0, 0]]),
                np.array([[0, 1], [1, 0]]),
                np.array([[1, 0], [0, 1]]),
                np.array([[1, 0], [1, 1]]),
                np.array([[1, 1], [0, 1]]),
            ]
        )
        self.cayley_db.database = database
        self.cayley_db.augment_by_equivalent_tables()
        self.assertTrue(np.allclose(true_database, self.cayley_db.database))

    def test_train_test_split(self):
        train, validation, test = self.cayley_db.train_test_split(2, 1)
        self.assertTrue(
            np.allclose(train.database, self.cayley_db.database[[2, 1]])
        )
        self.assertTrue(np.allclose(train.labels, [2, 1]))
        self.assertTrue(
            np.allclose(validation.database, self.cayley_db.database[3])
        )
        self.assertTrue(np.allclose(validation.labels, [3]))
        self.assertTrue(np.allclose(test.database, self.cayley_db.database[0]))
        self.assertTrue(np.allclose(test.labels, [0]))

    @patch("torch.load")
    def test_load_model(self, load_mock):
        self.cayley_db.load_model("model")
        load_mock.assert_called_once()

    def test_model(self):
        with self.assertRaises(ValueError):
            self.cayley_db.model

    def test_testing_report(self):
        self.cayley_db.model = lambda x: x
        np.random.seed(777)
        self.assertTrue(
            np.allclose(
                self.cayley_db.testing_report, [[4, 4], [1, 0], [1, 1]]
            )
        )
