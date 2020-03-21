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
from typing import no_type_check

import torch
from torch import Tensor
from torch.functional import einsum
from torch.nn import Module
from torch.nn.functional import kl_div


class AssociatorLoss(Module):
    """
    probabilistic associator loss
    """
    # pylint: disable=arguments-differ
    @no_type_check
    def forward(self, cayley_cube: Tensor) -> Tensor:
        """
        finds a probabilistic associator of a given probabilistic Cayley cube

:math:`P\\left\\{e_i\\left(e_je_k\\right)=e_l\\right\\}=
\\sum\\limits_{m=1}^nP\\left\\{e_i\\left(e_je_k\\right)=e_l\\vert
e_je_k=e_m\\right\\}P\\left\\{e_je_k=e_m\\right\\}=
\\sum\\limits_{m=1}^na_{iml}a_{jkm}`

        :param cayley_cube: this is a probabilistic Cayley cube
        :returns: the probabilistic associator
        """
        one = einsum("biml,bjkm->bijkl", cayley_cube, cayley_cube)
        two = einsum("bmkl,bijm->bijkl", cayley_cube, cayley_cube)
        return (kl_div(torch.log(one), two, reduction="sum") /
                cayley_cube.shape[0])
