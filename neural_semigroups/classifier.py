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
from typing import List, no_type_check

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
        :param hidden_dims: a list of sizes of hidden layers of a feed-forward network
        """
        super().__init__()
        self.cardinality = cardinality
        self.input_dim = cardinality ** 3
        hidden_layers: "OrderedDict[str, Module]" = OrderedDict()
        dims = [self.input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            hidden_layers.update(
                {f"linear{i}": Linear(dims[i], dims[i + 1], bias=True)}
            )
            hidden_layers.update({f"relu{i}": ReLU()})
            hidden_layers.update({f"bn{i}": BatchNorm1d(dims[i + 1])})
        self.hidden_layers = Sequential(hidden_layers)
        top_layers: "OrderedDict[str, Module]" = OrderedDict()
        top_layers.update(
            {f"linear{len(dims)}": Linear(dims[-1], 2, bias=False)}
        )
        top_layers.update({f"sigmoid": Sigmoid()})
        self.top_layers = Sequential(top_layers)

    # pylint: disable=arguments-differ
    @no_type_check
    def forward(self, cayley_cube: Tensor) -> Tensor:
        """
        forward pass inhereted from Module

        :param cayley_cube: probabilistic representation of a magma
        :returns: score of associativity (from ``0`` to ``1``)
        """
        last_hidden_layer = self.hidden_layers(
            cayley_cube.view(-1, self.input_dim)
        )
        return self.top_layers(last_hidden_layer)
