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

import numpy as np
import torch

from neural_semigroups.denoising_autoencoder import MagmaDAE
from neural_semigroups.magma import Magma
from neural_semigroups.utils import FOUR_GROUP


class TestMagmaDAE(TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cardinality = 4
        self.magma_dae = MagmaDAE(
            cardinality=self.cardinality,
            hidden_dims=[4],
            corruption_rate=0.6
        )

    def test_corruption(self):
        # first dimension is a batch size
        # for all x and y: x * y = 0
        cayley_cube = np.zeros([1, 4, 4, 4])
        cayley_cube[:, :, :, 0] = 1.0
        cayley_cube = torch.from_numpy(cayley_cube)
        self.magma_dae.train()
        true_value = np.array([
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25]
                ],
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25]
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25]
                ],
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25]
                ]
            ]
        ])
        self.assertTrue(np.allclose(
            self.magma_dae.corrupt_input(cayley_cube).numpy(),
            true_value
        ))
        self.magma_dae.apply_corruption = False
        self.assertTrue(np.allclose(
            self.magma_dae.corrupt_input(cayley_cube).numpy(),
            cayley_cube.numpy()
        ))

    def test_forward(self):
        cayley_cube = torch.from_numpy(
            np.stack([
                Magma(FOUR_GROUP).probabilistic_cube,
                Magma(FOUR_GROUP).probabilistic_cube
            ])
        ).view(-1, 4, 4, 4)
        true_value = np.array([
            [
                [
                    [0.13466279, 0.36468092, 0.3659935, 0.13466279],
                    [0.1344803, 0.3655452, 0.3654942, 0.1344803],
                    [0.17488064, 0.17488064, 0.4753581, 0.17488064],
                    [0.36552483, 0.13447367, 0.36552784, 0.13447367]
                ],
                [
                    [0.13548903, 0.3681852, 0.13548903, 0.36083674],
                    [0.3655246, 0.36552653, 0.13447446, 0.13447446],
                    [0.35086694, 0.13765107, 0.3738309, 0.13765107],
                    [0.13447313, 0.13447313, 0.36552268, 0.36553106]
                ],
                [
                    [0.36545593, 0.13449226, 0.13449226, 0.36555955],
                    [0.3655124, 0.36553752, 0.13447504, 0.13447504],
                    [0.29693836, 0.29694873, 0.2968655, 0.10924741],
                    [0.13449994, 0.3655188, 0.3654813, 0.13449994]],
                [
                    [0.36555028, 0.36548886, 0.13448043, 0.13448043],
                    [0.13559525, 0.3685539, 0.3602556, 0.13559525],
                    [0.10930068, 0.29684687, 0.29674882, 0.29710364],
                    [0.36552346, 0.13447334, 0.36552987, 0.13447334]
                ]
            ],
            [
                [
                    [0.36559576, 0.13449678,  0.13449678, 0.3654107],
                    [0.36554793, 0.13447915, 0.13447915, 0.36549377],
                    [0.29687777, 0.29689872, 0.1092574, 0.2969661],
                    [0.13679357, 0.37169805, 0.13679357, 0.35471484]
                ],
                [
                    [0.365516, 0.13448502, 0.36551395, 0.13448502],
                    [0.1345052, 0.1345052, 0.3654918, 0.3654978],
                    [0.1344896, 0.36556327, 0.1344896, 0.36545753],
                    [0.36545083, 0.3655707, 0.13448925, 0.13448925]
                ],
                [
                    [0.13485129, 0.36653146, 0.36376595, 0.13485129],
                    [0.13454393, 0.13454393, 0.36571753, 0.36519462],
                    [0.17488296, 0.17488296, 0.17488296, 0.4753511],
                    [0.3652153, 0.13453901, 0.13453901, 0.36570668]
                ],
                [
                    [0.13447537, 0.13447537, 0.3655291, 0.36552015],
                    [0.3655014, 0.13447976, 0.13447976, 0.36553907],
                    [0.47529095, 0.174903, 0.174903, 0.174903],
                    [0.13452795, 0.3655838, 0.13452795, 0.3653603]
                ]
            ]
        ])
        self.assertTrue(np.allclose(
            self.magma_dae(cayley_cube).detach().numpy(),
            true_value
        ))
