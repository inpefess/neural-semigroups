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

    def test_forward(self):
        cayley_cube = torch.stack(
            [
                Magma(FOUR_GROUP).probabilistic_cube,
                Magma(FOUR_GROUP).probabilistic_cube,
            ]
        ).view(-1, 4, 4, 4)
        true_value = torch.tensor(
            [
                [
                    [
                        [
                            7.1075952e-01,
                            9.6697964e-02,
                            9.6298940e-02,
                            9.6243583e-02,
                        ],
                        [
                            7.2343059e-02,
                            1.9661382e-01,
                            5.3442931e-01,
                            1.9661382e-01,
                        ],
                        [
                            2.2419298e-01,
                            8.2505047e-02,
                            6.0932779e-01,
                            8.3974160e-02,
                        ],
                        [
                            1.0000000e-06,
                            1.0000000e-06,
                            1.0000000e-06,
                            9.9999702e-01,
                        ],
                    ],
                    [
                        [
                            1.0000000e-06,
                            9.9999702e-01,
                            1.0000000e-06,
                            1.0000000e-06,
                        ],
                        [
                            9.9999702e-01,
                            1.0000000e-06,
                            1.0000000e-06,
                            1.0000000e-06,
                        ],
                        [
                            5.3320295e-01,
                            7.2544269e-02,
                            1.9712639e-01,
                            1.9712639e-01,
                        ],
                        [
                            5.4069214e-02,
                            3.9948660e-01,
                            1.4696524e-01,
                            3.9947897e-01,
                        ],
                    ],
                    [
                        [
                            1.7573349e-01,
                            4.7279951e-01,
                            1.7573349e-01,
                            1.7573349e-01,
                        ],
                        [
                            1.0000000e-06,
                            1.0000000e-06,
                            1.0000000e-06,
                            9.9999702e-01,
                        ],
                        [
                            9.9999702e-01,
                            1.0000000e-06,
                            1.0000000e-06,
                            1.0000000e-06,
                        ],
                        [
                            1.0000000e-06,
                            9.9999702e-01,
                            1.0000000e-06,
                            1.0000000e-06,
                        ],
                    ],
                    [
                        [
                            3.9949605e-01,
                            3.9940903e-01,
                            5.4087590e-02,
                            1.4700735e-01,
                        ],
                        [
                            1.0000000e-06,
                            1.0000000e-06,
                            9.9999702e-01,
                            1.0000000e-06,
                        ],
                        [
                            1.0000000e-06,
                            9.9999702e-01,
                            1.0000000e-06,
                            1.0000000e-06,
                        ],
                        [
                            3.9928460e-01,
                            1.4690571e-01,
                            3.9931062e-01,
                            5.4499064e-02,
                        ],
                    ],
                ],
                [
                    [
                        [
                            9.9999702e-01,
                            1.0000000e-06,
                            1.0000000e-06,
                            1.0000000e-06,
                        ],
                        [
                            1.0000000e-06,
                            9.9999702e-01,
                            1.0000000e-06,
                            1.0000000e-06,
                        ],
                        [
                            1.4803752e-01,
                            4.0226594e-01,
                            5.4468166e-02,
                            3.9522839e-01,
                        ],
                        [
                            6.0977799e-01,
                            2.2462027e-01,
                            8.2919694e-02,
                            8.2682036e-02,
                        ],
                    ],
                    [
                        [
                            1.0000000e-06,
                            9.9999702e-01,
                            1.0000000e-06,
                            1.0000000e-06,
                        ],
                        [
                            4.7532314e-01,
                            1.7489453e-01,
                            1.7490403e-01,
                            1.7487830e-01,
                        ],
                        [
                            7.2675504e-02,
                            5.3416759e-01,
                            1.9657844e-01,
                            1.9657844e-01,
                        ],
                        [
                            1.0000000e-06,
                            1.0000000e-06,
                            9.9999702e-01,
                            1.0000000e-06,
                        ],
                    ],
                    [
                        [
                            2.9658744e-01,
                            1.1023773e-01,
                            2.9658744e-01,
                            2.9658744e-01,
                        ],
                        [
                            3.9849406e-01,
                            1.4724270e-01,
                            4.0007952e-01,
                            5.4183703e-02,
                        ],
                        [
                            9.9999702e-01,
                            1.0000000e-06,
                            1.0000000e-06,
                            1.0000000e-06,
                        ],
                        [
                            1.0000000e-06,
                            9.9999702e-01,
                            1.0000000e-06,
                            1.0000000e-06,
                        ],
                    ],
                    [
                        [
                            1.0000000e-06,
                            1.0000000e-06,
                            1.0000000e-06,
                            9.9999702e-01,
                        ],
                        [
                            6.0671058e-02,
                            4.4825482e-01,
                            6.0668718e-02,
                            4.3040541e-01,
                        ],
                        [
                            1.0000000e-06,
                            9.9999702e-01,
                            1.0000000e-06,
                            1.0000000e-06,
                        ],
                        [
                            8.3026499e-02,
                            2.2566311e-01,
                            8.3021075e-02,
                            6.0828930e-01,
                        ],
                    ],
                ],
            ]
        )
        self.assertTrue(
            torch.allclose(self.magma_dae(cayley_cube).detach(), true_value)
        )
