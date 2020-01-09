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
from collections import OrderedDict
from typing import List

from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential, Sigmoid


class MagmaClassifier(Module):
    """
    Feed Forward Network for probability Cayley cubes of magmas
    """
    def __init__(
            self,
            cardinality: int,
            hidden_dims: List[int]
    ):
        """
        :param cardinality: the number of elements in a magma
        :param hidden_dims: a list of sizes of hidden layers of a feed-forward
        network
        """
        super().__init__()
        self.cardinality = cardinality
        self.input_dim = cardinality ** 3
        net_layers: "OrderedDict[str, Module]" = OrderedDict()
        dims = [self.input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            net_layers.update(
                {f"linear{i}": Linear(dims[i], dims[i + 1], bias=True)}
            )
            net_layers.update({f"relu{i}": ReLU()})
            net_layers.update({f"bn{i}": BatchNorm1d(dims[i + 1])})
        last_layer = len(dims)
        net_layers.update(
            {f"linear{last_layer}": Linear(dims[-1], 1, bias=True)}
        )
        net_layers.update({f"sigmoid": Sigmoid()})
        self.net_layers = Sequential(net_layers)

    # pylint: disable=arguments-differ
    def forward(self, cayley_cube: Tensor) -> Tensor:
        """
        forward pass inhereted from Module

        :param cayley_cube: probabilistic representation of a magma
        :returns: score of associativity (from 0 to 1)
        """
        return self.net_layers(cayley_cube.view(-1, self.input_dim))
