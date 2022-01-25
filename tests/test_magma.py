"""
   Copyright 2019-2022 Boris Shminke

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
# pylint: disable-all
from unittest import TestCase

import torch

from neural_semigroups.magma import Magma
from neural_semigroups.utils import FOUR_GROUP, NON_ASSOCIATIVE_MAGMA


class TestMagma(TestCase):
    def setUp(self):
        torch.manual_seed(777)
        self.some_magma = Magma(NON_ASSOCIATIVE_MAGMA)
        self.klein_group = Magma(FOUR_GROUP)

    def test_is_associative(self):
        self.assertFalse(Magma(torch.tensor([[0, 0], [1, 0]])).is_associative)
        self.assertTrue(self.klein_group.is_associative)
        self.assertFalse(self.some_magma.is_associative)

    def test_is_commutative(self):
        self.assertTrue(self.klein_group.is_commutative)
        self.assertFalse(self.some_magma.is_commutative)

    def test_get_identity(self):
        self.assertEqual(self.klein_group.identity, 0)
        self.assertEqual(self.some_magma.identity, -1)

    def test_has_inverses(self):
        self.assertTrue(self.klein_group.has_inverses)
        self.assertFalse(self.some_magma.has_inverses)

    def test_probabilistic_cube(self):
        cube = self.klein_group.probabilistic_cube
        self.assertIsInstance(cube, torch.Tensor)
        self.assertEqual(cube.shape, (4, 4, 4))
        self.assertEqual(cube.dtype, torch.float32)
        self.assertTrue(
            torch.allclose(
                cube,
                torch.tensor(
                    [
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        [
                            [0.0, 1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                ),
            )
        )
