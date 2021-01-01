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

from neural_semigroups.magma import Magma


class CyclicGroup(Magma):
    """
    finite cyclic group
    """

    def __init__(self, cardinality: int):
        """
        :param cardinality: number of elements in a cyclic group
        """
        identity_row = torch.arange(cardinality).reshape(cardinality, 1)
        super().__init__(
            (identity_row.transpose(0, 1) + identity_row) % cardinality
        )
