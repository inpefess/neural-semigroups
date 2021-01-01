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
import torch
from torch import Tensor
from torch.nn import Module

from neural_semigroups.constants import CURRENT_DEVICE


# pylint: disable=abstract-method
class ConstantBaseline(Module):
    """
    A model that always fills in the same number
    """

    def __init__(self, cardinality: int, fill_in_with: int = 0):
        """
        initializes a constant model

        >>> ConstantBaseline(2, 3)
        Traceback (most recent call last):
            ...
        ValueError: `fill_in_with` should be non-negative and less than `cardinality`, got 3 >= 2
        >>> ConstantBaseline(2, 1).constant_distribution.cpu()
        tensor([0., 1.])

        :param cardinality: the number of elements in a magma
        :param fill_in_with: an item which will be suggested as a correct answer
        """
        super().__init__()
        if fill_in_with >= cardinality:
            raise ValueError(
                "`fill_in_with` should be non-negative "
                "and less than `cardinality`, "
                f"got {fill_in_with} >= {cardinality}"
            )
        self.cardinality = cardinality
        self.constant_distribution = torch.zeros(
            cardinality, dtype=torch.float
        ).to(CURRENT_DEVICE)
        self.constant_distribution[fill_in_with] = 1.0

    # pylint: disable=arguments-differ,unused-argument
    def forward(self, cayley_cube: Tensor) -> Tensor:
        """
        forward pass inhereted from Module

        >>> ConstantBaseline(2, 1)(torch.tensor([
        ...     [[[0., 1.], [0.5, 0.5]], [[1., 0.], [0., 1.]]],
        ...     [[[0., 1.], [1.0, 0.0]], [[0.5, 0.5], [0., 1.]]]
        ... ]).to(CURRENT_DEVICE)).cpu()
        tensor([[[[0., 1.],
          [0., 1.]],
        <BLANKLINE>
         [[1., 0.],
          [0., 1.]]],
        <BLANKLINE>
        <BLANKLINE>
        [[[0., 1.],
          [1., 0.]],
        <BLANKLINE>
         [[0., 1.],
          [0., 1.]]]])

        :param cayley_cube: probabilistic representation of a magma
        :returns: a batch of constant values (set in the constructor)
        """
        result = cayley_cube.clone().detach()
        result[cayley_cube.max(3).values != 1.0] = self.constant_distribution
        return result
