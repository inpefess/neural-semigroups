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

from torch import from_numpy

from neural_semigroups.associator_loss import AssociatorLoss
from neural_semigroups.magma import Magma
from neural_semigroups.utils import FOUR_GROUP, NON_ASSOCIATIVE_MAGMA


class TestAssociatorLoss(TestCase):
    def test_forward(self):
        associator_loss = AssociatorLoss()
        cayley_cube = from_numpy(Magma(FOUR_GROUP).probabilistic_cube)
        self.assertEqual(associator_loss(cayley_cube), 0.0)
        cayley_cube = from_numpy(
            Magma(NON_ASSOCIATIVE_MAGMA).probabilistic_cube
        )
        self.assertEqual(associator_loss(cayley_cube), 9.0)
