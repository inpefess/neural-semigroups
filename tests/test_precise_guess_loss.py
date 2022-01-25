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

from neural_semigroups.constants import CURRENT_DEVICE
from neural_semigroups.cyclic_group import CyclicGroup
from neural_semigroups.magma import Magma
from neural_semigroups.precise_guess_loss import PreciseGuessLoss
from neural_semigroups.utils import FOUR_GROUP


class TestPreciseGuessLoss(TestCase):
    def test_forward(self):
        precise_guess_loss = PreciseGuessLoss()
        predicted_cubes = (
            torch.stack(
                [
                    Magma(FOUR_GROUP).probabilistic_cube,
                    CyclicGroup(4).probabilistic_cube,
                ]
            )
            .to(CURRENT_DEVICE)
            .view(-1, 4, 4, 4)
        )
        target_cubes = (
            torch.stack(
                [
                    Magma(FOUR_GROUP).probabilistic_cube,
                    Magma(FOUR_GROUP).probabilistic_cube,
                ]
            )
            .to(CURRENT_DEVICE)
            .view(-1, 4, 4, 4)
        )
        self.assertEqual(
            precise_guess_loss(predicted_cubes, target_cubes), 0.5
        )
        self.assertEqual(precise_guess_loss(target_cubes, target_cubes), 1.0)
