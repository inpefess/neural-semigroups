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
from torch import Tensor
from torch.nn import Module

from neural_semigroups.utils import count_different, make_discrete


# pylint: disable=abstract-method
class PreciseGuessLoss(Module):
    """
    loss for comparing probabilistic Cayley cubes precisely
    """

    # pylint: disable=arguments-differ, no-self-use
    def forward(self, predicted_cubes: Tensor, target_cubes: Tensor) -> Tensor:
        """
        finds a percentage of predicted Cayley tables,
        identical to the target ones
        """
        predicted_discrete = make_discrete(predicted_cubes)
        target_discrete = make_discrete(target_cubes)
        associator = count_different(predicted_discrete, target_discrete)
        return associator / predicted_cubes.shape[0]
