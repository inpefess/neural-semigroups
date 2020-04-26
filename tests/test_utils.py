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

import numpy as np

from neural_semigroups.utils import (
    FOUR_GROUP,
    check_filename,
    check_smallsemi_filename,
    get_anti_isomorphic_magmas,
    get_equivalent_magmas,
    get_isomorphic_magmas,
    get_magma_by_index,
    import_smallsemi_format,
    random_semigroup,
)


class TestUtils(TestCase):
    def setUp(self):
        np.random.seed(0)
        self.true_isomorphic_groups = [
            [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
            [[1, 0, 3, 2], [0, 1, 2, 3], [3, 2, 1, 0], [2, 3, 0, 1]],
            [[2, 3, 0, 1], [3, 2, 1, 0], [0, 1, 2, 3], [1, 0, 3, 2]],
            [[3, 2, 1, 0], [2, 3, 0, 1], [1, 0, 3, 2], [0, 1, 2, 3]],
        ]

    def test_random_semigroup(self):
        success, cayley_table = random_semigroup(2, 1)
        self.assertTrue(success)
        self.assertTrue(np.allclose(cayley_table, np.array([[0, 1], [1, 0]])))

    def test_check_filename(self):
        with self.assertRaises(Exception):
            check_filename(123)
        with self.assertRaises(Exception):
            check_filename("strange.name")
        with self.assertRaises(Exception):
            check_filename("very.strange.name")
        with self.assertRaises(Exception):
            check_filename("semigroup.strange.name")
        with self.assertRaises(Exception):
            check_filename("semigroup.name.npz")
        self.assertEqual(check_filename("monoid.1.npz"), 1)

    def test_check_smallsemi_filename(self):
        with self.assertRaises(Exception):
            check_smallsemi_filename(123)
        with self.assertRaises(Exception):
            check_smallsemi_filename("very.very.strange.name")
        with self.assertRaises(Exception):
            check_smallsemi_filename("very.strange.name")
        with self.assertRaises(Exception):
            check_smallsemi_filename("strange.name.gz")
        with self.assertRaises(Exception):
            check_smallsemi_filename("name.gl.gz")
        with self.assertRaises(Exception):
            check_smallsemi_filename("datan.gl.gz")
        with self.assertRaises(Exception):
            check_smallsemi_filename("data1.gl.gz")
        self.assertEqual(check_smallsemi_filename("data2.gl.gz"), 2)

    def test_get_magma_by_index(self):
        self.assertTrue(
            np.allclose(
                get_magma_by_index(2, 0).cayley_table,
                np.array([[0, 0], [0, 0]]),
            )
        )
        self.assertTrue(
            np.allclose(
                get_magma_by_index(2, 15).cayley_table,
                np.array([[1, 1], [1, 1]]),
            )
        )
        self.assertTrue(
            np.allclose(
                get_magma_by_index(2, 7).cayley_table,
                np.array([[0, 1], [1, 1]]),
            )
        )
        with self.assertRaises(Exception):
            get_magma_by_index(3, -1)
        with self.assertRaises(Exception):
            get_magma_by_index(3, 20000)

    def test_import_smallsemi_format(self):
        lines = [b"# header", b"0100\r", b"0101\r", b"0011\r"]
        semigroups = import_smallsemi_format(lines)
        self.assertIsInstance(semigroups, np.ndarray)
        self.assertEqual(semigroups.shape[0], 4)
        true_semigroups = [
            [[0, 0], [0, 0]],
            [[0, 1], [1, 0]],
            [[0, 0], [0, 1]],
            [[0, 0], [1, 1]],
        ]
        for i in range(4):
            self.assertIsInstance(semigroups[i], np.ndarray)
            self.assertTrue(
                np.allclose(semigroups[i], np.array(true_semigroups[i]))
            )

    def test_get_isomorphic_magmas(self):
        isomorphic_groups = get_isomorphic_magmas(FOUR_GROUP)
        self.assertIsInstance(isomorphic_groups, np.ndarray)
        n = isomorphic_groups.shape[0]
        self.assertEqual(n, 4)
        for i in range(n):
            self.assertIsInstance(isomorphic_groups[i], np.ndarray)
            self.assertTrue(
                np.allclose(
                    isomorphic_groups[i],
                    np.array(self.true_isomorphic_groups[i]),
                )
            )
        isomorphic_magmas = get_isomorphic_magmas(np.array([[1, 1], [0, 0]]))
        self.assertIsInstance(isomorphic_magmas, np.ndarray)
        n = isomorphic_magmas.shape[0]
        self.assertEqual(n, 1)
        isomorphic_magma = [[1, 1], [0, 0]]
        self.assertIsInstance(isomorphic_magmas[0], np.ndarray)
        self.assertTrue(np.allclose(isomorphic_magmas[0], isomorphic_magma))

    def test_get_anti_isomorphic_magmas(self):
        anti_isomorphic_groups = get_anti_isomorphic_magmas(FOUR_GROUP)
        self.assertIsInstance(anti_isomorphic_groups, np.ndarray)
        n = anti_isomorphic_groups.shape[0]
        self.assertEqual(n, 4)
        for i in range(n):
            self.assertIsInstance(anti_isomorphic_groups[i], np.ndarray)
            self.assertTrue(
                np.allclose(
                    anti_isomorphic_groups[i],
                    np.array(self.true_isomorphic_groups[i]),
                )
            )
        isomorphic_magmas = get_anti_isomorphic_magmas(
            np.array([[1, 1], [0, 0]])
        )
        self.assertIsInstance(isomorphic_magmas, np.ndarray)
        n = isomorphic_magmas.shape[0]
        self.assertEqual(n, 1)
        anti_isomorphic_magma = np.array([[1, 0], [1, 0]])
        self.assertIsInstance(isomorphic_magmas[0], np.ndarray)
        self.assertTrue(
            np.allclose(isomorphic_magmas[0], anti_isomorphic_magma)
        )

    def test_get_equivalent_magmas(self):
        equivalent_groups = get_equivalent_magmas(FOUR_GROUP)
        self.assertIsInstance(equivalent_groups, np.ndarray)
        n = equivalent_groups.shape[0]
        self.assertEqual(n, 4)
        for i in range(n):
            self.assertIsInstance(equivalent_groups[i], np.ndarray)
            self.assertTrue(
                np.allclose(
                    equivalent_groups[i],
                    np.array(self.true_isomorphic_groups[i]),
                )
            )
        equivalent_magmas = get_equivalent_magmas(np.array([[1, 1], [0, 0]]))
        true_equivalent_magmas = [[[1, 0], [1, 0]], [[1, 1], [0, 0]]]
        self.assertIsInstance(equivalent_magmas, np.ndarray)
        n = equivalent_magmas.shape[0]
        self.assertEqual(n, 2)
        for i in range(n):
            self.assertIsInstance(equivalent_magmas[i], np.ndarray)
            self.assertTrue(
                np.allclose(
                    equivalent_magmas[i], np.array(true_equivalent_magmas[i])
                )
            )
