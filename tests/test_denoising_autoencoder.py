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
        true_value = np.array(
            [
                [
                    [
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                        [1.0, 0.0, 0.0, 0.0]],
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25]
                    ],
                    [
                        [0.25, 0.25, 0.25, 0.25],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0]
                    ],
                    [
                        [0.25, 0.25, 0.25, 0.25],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.25, 0.25, 0.25, 0.25]
                    ]
                ]
            ]
        )
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
        true_value = np.array(
            [
                [
                    [
                        [0.7107595, 0.09669796, 0.09629894, 0.09624358],
                        [0.07234306, 0.19661382, 0.5344293, 0.19661382],
                        [0.22419298, 0.08250505, 0.6093278, 0.08397416],
                        [0.05421957, 0.14719021, 0.39872202, 0.3998682]
                    ],
                    [
                        [0.31880915, 0.04325254, 0.31853363, 0.3194047],
                        [0.10924885, 0.29691327, 0.29689708, 0.2969408],
                        [0.53320295, 0.07254427, 0.19712639, 0.19712639],
                        [0.05406921, 0.3994866, 0.14696524, 0.39947897]
                    ],
                    [
                        [0.17573349, 0.4727995, 0.17573349, 0.17573349],
                        [0.0829399, 0.22446671, 0.0826113, 0.6099821],
                        [0.22450335, 0.08259429, 0.610234, 0.08266836],
                        [0.08262523, 0.22453031, 0.6102334, 0.08261103]
                    ],
                    [
                        [0.39949605, 0.39940903, 0.05408759, 0.14700735],
                        [0.43929967, 0.05945898, 0.43931654, 0.06192481],
                        [0.39950228, 0.14698358, 0.05408172, 0.39943242],
                        [0.3992846, 0.1469057, 0.39931062, 0.05449906]
                    ]
                ],
                [
                    [
                        [0.04325994, 0.31797373, 0.31929135, 0.31947497],
                        [0.5344007, 0.19663005, 0.07233919, 0.19663005],
                        [0.14803752, 0.40226594, 0.05446817, 0.3952284],
                        [0.609778, 0.22462027, 0.08291969, 0.08268204]
                    ],
                    [
                        [0.09643315, 0.71079695, 0.09651656, 0.09625335],
                        [0.47532314, 0.17489453, 0.17490403, 0.1748783],
                        [0.0726755, 0.5341676, 0.19657844, 0.19657844],
                        [0.61027676, 0.08259898, 0.22452374, 0.08260055]
                    ],
                    [
                        [0.29658744, 0.11023773, 0.29658744, 0.29658744],
                        [0.39849406, 0.1472427, 0.40007952, 0.0541837],
                        [0.14702089, 0.3996243, 0.05408857, 0.39926624],
                        [0.39943075, 0.1469871, 0.05408268, 0.39949945]
                    ],
                    [
                        [0.08261844, 0.08263643, 0.6102275, 0.22451758],
                        [0.06067106, 0.44825482, 0.06066872, 0.4304054],
                        [0.08260918, 0.22453228, 0.6102349, 0.08262362],
                        [0.0830265, 0.22566311, 0.08302107, 0.6082893]
                    ]
                ]
            ]
        )
        self.assertTrue(np.allclose(
            self.magma_dae(cayley_cube).detach().numpy(),
            true_value
        ))
