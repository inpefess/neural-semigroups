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
# pylint: disable-all
from unittest import TestCase

import torch

from neural_semigroups.utils import (
    FOUR_GROUP,
    check_filename,
    check_smallsemi_filename,
    corrupt_input,
    get_equivalent_magmas,
    get_magma_by_index,
    import_smallsemi_format,
    random_semigroup,
)


class TestUtils(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.true_isomorphic_groups = torch.tensor(
            [
                [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
                [[1, 0, 3, 2], [0, 1, 2, 3], [3, 2, 1, 0], [2, 3, 0, 1]],
                [[2, 3, 0, 1], [3, 2, 1, 0], [0, 1, 2, 3], [1, 0, 3, 2]],
                [[3, 2, 1, 0], [2, 3, 0, 1], [1, 0, 3, 2], [0, 1, 2, 3]],
            ]
        )

    def test_random_semigroup(self):
        success, cayley_table = random_semigroup(2, 1)
        self.assertTrue(success)
        self.assertTrue(
            torch.allclose(cayley_table, torch.tensor([[0, 1], [1, 0]]))
        )

    def test_check_filename(self):
        with self.assertRaises(ValueError):
            check_filename(123)
        with self.assertRaises(ValueError):
            check_filename("strange.name")
        with self.assertRaises(ValueError):
            check_filename("very.strange.name")
        with self.assertRaises(ValueError):
            check_filename("semigroup.strange.name")
        with self.assertRaises(ValueError):
            check_filename("semigroup.name.zip")
        self.assertEqual(check_filename("monoid.1.zip"), 1)

    def test_check_smallsemi_filename(self):
        with self.assertRaises(ValueError):
            check_smallsemi_filename(123)
        with self.assertRaises(ValueError):
            check_smallsemi_filename("very.very.strange.name")
        with self.assertRaises(ValueError):
            check_smallsemi_filename("very.strange.name")
        with self.assertRaises(ValueError):
            check_smallsemi_filename("strange.name.gz")
        with self.assertRaises(ValueError):
            check_smallsemi_filename("name.gl.gz")
        with self.assertRaises(ValueError):
            check_smallsemi_filename("datan.gl.gz")
        with self.assertRaises(ValueError):
            check_smallsemi_filename("data1.gl.gz")
        self.assertEqual(check_smallsemi_filename("data2.gl.gz"), 2)

    def test_get_magma_by_index(self):
        self.assertTrue(
            torch.allclose(
                get_magma_by_index(2, 0).cayley_table,
                torch.tensor([[0, 0], [0, 0]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                get_magma_by_index(2, 15).cayley_table,
                torch.tensor([[1, 1], [1, 1]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                get_magma_by_index(2, 7).cayley_table,
                torch.tensor([[0, 1], [1, 1]]),
            )
        )
        with self.assertRaises(ValueError):
            get_magma_by_index(3, -1)
        with self.assertRaises(ValueError):
            get_magma_by_index(3, 20000)

    def test_import_smallsemi_format(self):
        lines = [b"# header", b"0100\r", b"0101\r", b"0011\r"]
        semigroups = import_smallsemi_format(lines)
        self.assertIsInstance(semigroups, torch.Tensor)
        self.assertEqual(semigroups.shape[0], 4)
        true_semigroups = [
            [[0, 0], [0, 0]],
            [[0, 1], [1, 0]],
            [[0, 0], [0, 1]],
            [[0, 0], [1, 1]],
        ]
        for i in range(4):
            self.assertIsInstance(semigroups[i], torch.Tensor)
            self.assertTrue(
                torch.allclose(semigroups[i], torch.tensor(true_semigroups[i]))
            )

    def test_get_equivalent_magmas(self):
        equivalent_groups = get_equivalent_magmas(FOUR_GROUP.view(1, 4, 4))
        self.assertIsInstance(equivalent_groups, torch.Tensor)
        n = equivalent_groups.shape[0]
        self.assertEqual(n, 4)
        for i in range(n):
            self.assertIsInstance(equivalent_groups[i], torch.Tensor)
            self.assertTrue(
                torch.allclose(
                    equivalent_groups[i], self.true_isomorphic_groups[i],
                )
            )
        equivalent_magmas = get_equivalent_magmas(
            torch.tensor([[[1, 1], [0, 0]]])
        )
        true_equivalent_magmas = [[[1, 0], [1, 0]], [[1, 1], [0, 0]]]
        self.assertIsInstance(equivalent_magmas, torch.Tensor)
        n = equivalent_magmas.shape[0]
        self.assertEqual(n, 2)
        for i in range(n):
            self.assertIsInstance(equivalent_magmas[i], torch.Tensor)
            self.assertTrue(
                torch.allclose(
                    equivalent_magmas[i],
                    torch.tensor(true_equivalent_magmas[i]),
                )
            )

    def test_dropout(self):
        # first dimension is a batch size
        # for all x and y: x * y = 0
        cayley_cube = torch.zeros([1, 4, 4, 4])
        cayley_cube[:, :, :, 0] = 1.0
        true_value = torch.tensor(
            [
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [1.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [1.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.25, 0.25, 0.25, 0.25],
                    ],
                    [
                        [0.25, 0.25, 0.25, 0.25],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ]
        )
        self.assertTrue(
            torch.allclose(corrupt_input(cayley_cube, 0.5), true_value)
        )
        self.assertTrue(
            torch.allclose(corrupt_input(cayley_cube, 0.0), cayley_cube,)
        )
