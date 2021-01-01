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
from torch.functional import einsum
from torch.nn import Module
from torch.nn.functional import kl_div

from neural_semigroups.utils import count_different, make_discrete


# pylint: disable=abstract-method
class AssociatorLoss(Module):
    """
    probabilistic associator loss
    """

    def __init__(self, discrete: bool = False):
        """
        :param discrete: when ``False``, the KL divergence is is used for measuring associativity in a continuous way.
            when ``True``, returns ``1`` for associative and ``0`` for non associative samples.
        """
        super().__init__()
        self.discrete = discrete

    # pylint: disable=arguments-differ
    def forward(self, cayley_cubes: Tensor) -> Tensor:
        """
        finds a probabilistic associator of a given probabilistic Cayley cube

        First, we build two 4-index tensors representing probability distributions of products :math:`a\\left(bc\\right)`
        and :math:`\\left(ab\\right)c`, respectively:

        :math:`T_{ijkl}=P\\left\\{e_i\\left(e_je_k\\right)=e_l\\right\\}=
        \\sum\\limits_{m=1}^nP\\left\\{e_ie_m=e_l\\vert e_je_k=e_m\\right\\}
        P\\left\\{e_je_k=e_m\\right\\}=\\sum\\limits_{m=1}^na_{iml}a_{jkm}`

        and

        :math:`T\\prime_{ijkl}=P\\left\\{\\left(e_ie_j\\right)e_k=e_l\\right\\}=
        \\sum\\limits_{m=1}^nP\\left\\{e_me_k=e_l\\vert e_ie_j=e_m\\right\\}
        P\\left\\{e_ie_j=e_m\\right\\}=\\sum\\limits_{m=1}^na_{mkl}a_{ijm}`

        Then we calculate `Kullback-Leibler divergence`_ between :math:`T_{ijkl}`
        and :math:`T\\prime_{ijkl}` to find a continuous measure of associativity of the input table.

        :param cayley_cubes: a batch of probabilistic Cayley cubes
        :returns: the probabilistic associator

        .. _Kullback-Leibler divergence: https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

        """
        cubes = make_discrete(cayley_cubes) if self.discrete else cayley_cubes
        one = einsum("biml,bjkm->bijkl", cubes, cubes)
        two = einsum("bmkl,bijm->bijkl", cubes, cubes)
        batch_size = cayley_cubes.shape[0]
        associator = (
            count_different(one, two)
            if self.discrete
            else kl_div(torch.log(one), two, reduction="sum")
        )
        return associator / batch_size
