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

from neural_semigroups.constants import CURRENT_DEVICE
from neural_semigroups.denoising_autoencoder import MagmaDAE
from neural_semigroups.magma import Magma
from neural_semigroups.utils import FOUR_GROUP


class TestMagmaDAE(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cardinality = 4
        self.magma_dae = MagmaDAE(
            cardinality=self.cardinality, hidden_dims=[4], dropout_rate=0.6
        ).to(CURRENT_DEVICE)
        self.magma_vae = MagmaDAE(
            cardinality=self.cardinality,
            hidden_dims=[4],
            dropout_rate=0.6,
            do_reparametrization=True,
        )

    def test_forward(self):
        cayley_cube = (
            torch.stack(
                [
                    Magma(FOUR_GROUP).probabilistic_cube,
                    Magma(FOUR_GROUP).probabilistic_cube,
                ]
            )
            .view(-1, 4, 4, 4)
            .to(CURRENT_DEVICE)
        )
        self.assertEqual(
            self.magma_dae(cayley_cube).detach().sum().item(), 32.0,
        )
        self.assertEqual(
            self.magma_vae(cayley_cube).detach().sum().item(), 32.0,
        )
