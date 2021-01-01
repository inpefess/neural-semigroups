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
# pylint: disable-all
from unittest import TestCase

import torch

from neural_semigroups.cyclic_group import CyclicGroup


class TestCyclicGroup(TestCase):
    def test_cyclic_group(self):
        cyclic_group = CyclicGroup(4)
        self.assertTrue(
            cyclic_group.cayley_table.allclose(
                torch.tensor(
                    [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
                )
            )
        )
        self.assertTrue(cyclic_group.is_associative)
        self.assertTrue(cyclic_group.is_commutative)
        self.assertEqual(cyclic_group.identity, 0)
