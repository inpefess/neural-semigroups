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
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from neural_semigroups.cayley_database import CayleyDatabase, train_test_split


class TestCayleyDatabase(TestCase):
    def setUp(self):
        np.random.seed(43)
        self.cayley_db = CayleyDatabase()
        self.cayley_db.database = np.array([
            np.array([[0, 1], [1, 0]]), np.array([[1, 0], [1, 1]]),
            np.array([[1, 1], [1, 0]]), np.array([[1, 1], [1, 1]])
        ])

    @patch("numpy.load")
    def test_load_database(self, numpy_load_mock):
        database = np.array([[[0, 1], [2, 3]], [[1, 0], [3, 2]]])
        npz_file = MagicMock()
        npz_file.__getitem__ = lambda x, y: database
        numpy_load_mock.return_value = npz_file
        with patch.object(npz_file, "close") as mock:
            self.cayley_db.load_database("semigroup.2.npz")
            mock.assert_called_once()
        self.assertEqual(self.cayley_db.cardinality, 2)
        self.assertTrue(np.allclose(self.cayley_db.database, database))

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

    def test_predict_from_model(self):
        self.cayley_db.cardinality = 2
        input = [[-1, 0], [0, 1]]
        table, cube = self.cayley_db.predict_from_model(input)
        self.assertIsNone(table)
        self.assertIsNone(cube)
        self.cayley_db.model = lambda x: x
        table, cube = self.cayley_db.predict_from_model(input)
        self.assertIsInstance(table, np.ndarray)
        self.assertIsInstance(cube, np.ndarray)
        self.assertEqual(table.dtype, int)
        self.assertEqual(cube.dtype, np.float32)
        self.assertTrue(np.allclose(table, np.array([[0, 0], [0, 1]])))
        self.assertTrue(np.allclose(
            cube,
            np.array([
                [[0.5, 0.5], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]]
            ])
        ))
        with self.assertRaises(Exception):
            self.cayley_db.predict_from_model("no good")

    def test_check_input(self):
        self.cayley_db.cardinality = 2
        self.assertFalse(self.cayley_db._check_input([[0]]))
        self.assertFalse(self.cayley_db._check_input([[0.5, 0], [0, 0]]))
        self.assertFalse(self.cayley_db._check_input([[-2, 0], [0, 0]]))
        self.assertFalse(self.cayley_db._check_input([[2, 0], [0, 0]]))

    def test_augment_by_equivalent_tables(self):
        database = np.array([
            np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]]),
            np.array([[1, 0], [1, 1]])
        ])
        true_database = np.array([
            np.array([[0, 0], [1, 0]]), np.array([[0, 1], [0, 0]]),
            np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]]),
            np.array([[1, 0], [1, 1]]), np.array([[1, 1], [0, 1]])
        ])
        self.cayley_db.database = database
        self.cayley_db.augment_by_equivalent_tables()
        self.assertTrue(np.allclose(
            true_database, self.cayley_db.database
        ))

    def test_train_test_split(self):
        train, validation, test = train_test_split(self.cayley_db, 2, 1)
        self.assertTrue(np.allclose(
            train.database, self.cayley_db.database[[1, 2]]
        ))
        self.assertTrue(np.allclose(
            validation.database, self.cayley_db.database[0]
        ))
        self.assertTrue(np.allclose(
            test.database, self.cayley_db.database[3]
        ))

    @patch("builtins.open", mock_open())
    @patch("neural_semigroups.cayley_database.import_smallsemi_format")
    def test_load_smallsemi_database(self, import_smallsemi_format_mock):
        self.cayley_db.load_smallsemi_database("data1.gl")
        import_smallsemi_format_mock.assert_called_once()

    @patch("torch.load")
    def test_load_model(self, load_mock):
        self.cayley_db.load_model("model")
        load_mock.assert_called_once()
