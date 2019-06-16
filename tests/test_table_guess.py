"""
   Copyright 2019 Boris Shminke

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
from unittest.mock import mock_open, patch

import numpy as np

from neural_semigroups.table_guess import TableGuess, train_test_split


class TestTableGuess(TestCase):
    def setUp(self):
        np.random.seed(43)
        self.table_guess = TableGuess()
        self.table_guess.database = np.array([
            np.array([[0, 1], [1, 0]]), np.array([[1, 0], [1, 1]]),
            np.array([[1, 1], [1, 0]]), np.array([[1, 1], [1, 1]])
        ])

    @patch("builtins.open", mock_open(read_data="0 1 2 3\n1 0 3 2"))
    def test_load_database(self):
        self.table_guess.load_database("semigroup.2.dat")
        self.assertEqual(self.table_guess.cardinality, 2)
        database = [np.array([[0, 1], [2, 3]]), np.array([[1, 0], [3, 2]])]
        for i, table in enumerate(database):
            self.assertTrue(np.allclose(
                self.table_guess.database[i],
                table
            ))

    def test_search_database(self):
        self.table_guess.cardinality = 2
        complete = self.table_guess.search_database([[-1, 1], [1, 0]])
        self.assertIsInstance(complete, list)
        self.assertEqual(len(complete), 2)
        self.assertTrue(np.allclose(complete[0], self.table_guess.database[0]))
        self.assertTrue(np.allclose(complete[1], self.table_guess.database[2]))
        complete = self.table_guess.search_database([[-1, -1], [0, 0]])
        self.assertIsInstance(complete, list)
        self.assertEqual(len(complete), 0)
        with self.assertRaises(Exception):
            self.table_guess.search_database("no good")

    def test_predict_from_model(self):
        self.table_guess.cardinality = 2
        input = [[-1, 0], [0, 1]]
        table, cube = self.table_guess.predict_from_model(input)
        self.assertIsNone(table)
        self.assertIsNone(cube)
        self.table_guess.model = lambda x: x
        table, cube = self.table_guess.predict_from_model(input)
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
            self.table_guess.predict_from_model("no good")

    def test_check_input(self):
        self.table_guess.cardinality = 2
        self.assertFalse(self.table_guess._check_input([[0]]))
        self.assertFalse(self.table_guess._check_input([[0.5, 0], [0, 0]]))
        self.assertFalse(self.table_guess._check_input([[-2, 0], [0, 0]]))
        self.assertFalse(self.table_guess._check_input([[2, 0], [0, 0]]))

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
        self.table_guess.database = database
        self.table_guess.augment_by_equivalent_tables()
        self.assertTrue(np.allclose(
            true_database, self.table_guess.database
        ))

    def test_train_test_split(self):
        train, validation, test = train_test_split(self.table_guess, 2, 1)
        self.assertTrue(np.allclose(
            train.database, self.table_guess.database[[1, 2]]
        ))
        self.assertTrue(np.allclose(
            validation.database, self.table_guess.database[0]
        ))
        self.assertTrue(np.allclose(
            test.database, self.table_guess.database[3]
        ))

    @patch("builtins.open", mock_open())
    @patch("neural_semigroups.table_guess.import_smallsemi_format")
    def test_load_smallsemi_database(self, import_smallsemi_format_mock):
        self.table_guess.load_smallsemi_database("data1.gl")
        import_smallsemi_format_mock.assert_called_once()

    @patch("torch.load")
    def test_load_model(self, load_mock):
        self.table_guess.load_model("model")
        load_mock.assert_called_once()
