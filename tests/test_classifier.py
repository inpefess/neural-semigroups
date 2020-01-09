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
import torch

from neural_semigroups.classifier import MagmaClassifier
from neural_semigroups.magma import Magma
from neural_semigroups.utils import FOUR_GROUP


class TestMagmaClassifier(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cardinality = 4
        self.classifier = MagmaClassifier(
            cardinality=self.cardinality,
            hidden_dims=[4]
        )

    def test_forward(self):
        cayley_cube = torch.from_numpy(
            np.stack([
                Magma(FOUR_GROUP).probabilistic_cube,
                Magma(FOUR_GROUP).probabilistic_cube
            ])
        ).view(-1, 4, 4, 4)
        true_value = np.array([[0.5452165], [0.5452165]])
        self.assertTrue(np.allclose(
            self.classifier(cayley_cube).detach().numpy(),
            true_value
        ))
