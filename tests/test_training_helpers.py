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
import sys
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from neural_semigroups.training_helpers import (get_arguments, get_loaders,
                                                load_database_as_cubes)


class TestTrainingHelpers(TestCase):
    def test_get_arguments(self):
        args = "prog --cardinality 2 --train_size 1 --validation_size 1"
        with patch.object(sys, "argv", args.split(" ")):
            get_arguments()

    @patch("neural_semigroups.training_helpers.load_database_as_cubes")
    def test_get_loaders(self, load_database_as_cubes_mock):
        train = np.array([1])
        train_labels = np.array([2])
        validation = np.array([3])
        validation_labels = np.array([4])
        load_database_as_cubes_mock.return_value = (
            train, validation, None, train_labels, validation_labels, None
        )
        train_loader, validation_loader = get_loaders(
            "filename", 1, 1, 1, True
        )
        batch = [i for i in train_loader][0]
        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), 2)
        self.assertTrue(np.allclose(batch[0].numpy(), train))
        self.assertTrue(np.allclose(batch[1].numpy(), train_labels))
        batch = [i for i in validation_loader][0]
        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), 2)
        self.assertTrue(np.allclose(batch[0].numpy(), validation))
        self.assertTrue(np.allclose(batch[1].numpy(), validation_labels))
        train_loader, validation_loader = get_loaders(
            "filename", 1, 1, 1, False
        )
        batch = [i for i in train_loader][0]
        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), 2)
        self.assertTrue(np.allclose(batch[0].numpy(), train))
        self.assertTrue(np.allclose(batch[1].numpy(), train))
        batch = [i for i in validation_loader][0]
        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), 2)
        self.assertTrue(np.allclose(batch[0].numpy(), validation))
        self.assertTrue(np.allclose(batch[1].numpy(), validation))