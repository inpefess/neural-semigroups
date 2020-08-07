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
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential, Softmax2d
from torch.nn.functional import dropout2d

from neural_semigroups.constants import CURRENT_DEVICE


def get_encoder_and_decoder_layers(
    dims: List[int], split_last: bool
) -> Tuple[Module, Module]:
    """
    construct symmetrical encoder and decoder modules

    :param dims: the dimensions of layers in an encoder part (the same dimensions are used for a decoder)
    :param split_last: if ``True``, makes their last encoder dimensions twice larger than the first decoder dimension (for a reparameterization trick)
    :returns: a pair of two sequential models, representing encoder and decoder layers
    """
    encoder_layers: "OrderedDict[str, Module]" = OrderedDict()
    decoder_layers: "OrderedDict[str, Module]" = OrderedDict()
    encoder_dims = dims.copy()
    decoder_dims = list(reversed(dims))
    if split_last:
        encoder_dims[-1] *= 2
    for i in range(len(dims) - 1):
        encoder_layers.update(
            {
                f"linear0{i}": Linear(
                    encoder_dims[i], encoder_dims[i + 1], bias=True
                )
            }
        )
        encoder_layers.update({f"relu0{i}": ReLU()})
        encoder_layers.update({f"bn0{i}": BatchNorm1d(encoder_dims[i + 1])})
        decoder_layers.update(
            {
                f"linear1{i}": Linear(
                    decoder_dims[i], decoder_dims[i + 1], bias=True
                )
            }
        )
        decoder_layers.update({f"relu1{i}": ReLU()})
        decoder_layers.update({f"bn1{i}": BatchNorm1d(decoder_dims[i + 1])})
    return (
        Sequential(encoder_layers).to(CURRENT_DEVICE),
        Sequential(decoder_layers).to(CURRENT_DEVICE),
    )


# pylint: disable=abstract-method
class MagmaDAE(Module):
    """
    Denoising Autoencoder for probability Cayley cubes of magmas
    """

    apply_corruption: bool = True

    def __init__(
        self,
        cardinality: int,
        hidden_dims: List[int],
        corruption_rate: float,
        do_reparametrization: bool = False,
    ):
        """
        :param cardinality: the number of elements in a magma
        :param hidden_dims: a list of sizes of hidden layers of the encoder and the decoder
        :param corruption_rate: what percentage of cells from the Cayley table to substitute with uniform random variables
        :param do_reparametrization: if ``True``, adds a reparameterization trick
        """
        super().__init__()
        self.cardinality = cardinality
        self.corruption_rate = corruption_rate
        self.do_reparametrization = do_reparametrization
        (
            self.encoder_layers,
            self.decoder_layers,
        ) = get_encoder_and_decoder_layers(
            [cardinality ** 3] + hidden_dims, do_reparametrization
        )
        # pylint: disable=not-callable
        self._nearly_zero = torch.tensor([1e-6], device=CURRENT_DEVICE)
        # pylint: disable=not-callable
        self._nearly_one = torch.tensor(
            [1 - (self.cardinality - 1) * 1e-6], device=CURRENT_DEVICE
        )

    def encode(self, corrupted_input: Tensor) -> Tensor:
        """
        represent input cube as an embedding vector

        :param corrupted_input: a tensor with two indices
        :returns: some tensor with two indices and non-negative values
        """
        return self.encoder_layers(
            corrupted_input.view(corrupted_input.shape[0], -1)
        )

    def corrupt_input(self, cayley_cube: Tensor) -> Tensor:
        """
        changes several cells in a Cayley table with uniformly distributed
        random variables

        :param cayley_cube: representation of a Cayley table probability distribution
        :returns: distorted Cayley cube
        """
        dropout_norm = (
            1 - self.corruption_rate if self.apply_corruption else 1.0
        )
        return (
            dropout_norm
            * dropout2d(
                cayley_cube.view(
                    -1,
                    self.cardinality * self.cardinality,
                    self.cardinality,
                    1,
                )
                - 1 / self.cardinality,
                self.corruption_rate,
                self.apply_corruption,
            )
            + 1 / self.cardinality
        ).view(-1, self.cardinality, self.cardinality, self.cardinality)

    def decode(self, encoded_input: Tensor) -> Tensor:
        """
        represent an embedding vector as something with size aligned with the
        input

        :param encoded_input: an embedding vector
        :returns: a vector of values from ``0`` to ``1`` (kind of probabilities)
        """
        return (
            Softmax2d()(
                self.decoder_layers(encoded_input)
                .view(-1, self.cardinality, self.cardinality, self.cardinality)
                .transpose(1, 3)
                .transpose(2, 3)
            )
            .transpose(2, 3)
            .transpose(1, 3)
        )

    def reparameterize(self, mu_and_sigma: Tensor) -> Tensor:
        """
        do a reparametrization trick

        :param mu_and_sigma: vector of expectation and standard deviation
        :returns: sample from a distribution
        """
        if self.do_reparametrization:
            dim = mu_and_sigma.shape[1] // 2
            sample = torch.normal(mu_and_sigma[:, :dim], mu_and_sigma[:, dim:])
        else:
            sample = mu_and_sigma
        return sample

    # pylint: disable=arguments-differ
    def forward(self, cayley_cube: Tensor) -> Tensor:
        """
        forward pass inhereted from Module

        :param cayley_cube: probabilistic representation of a magma
        :returns: autoencoded probabilistic representation of a magma
        """
        corrupted_input = self.corrupt_input(cayley_cube)
        encoded_input = self.encode(corrupted_input)
        reparameterized_input = self.reparameterize(encoded_input)
        decoded_input = self.decode(reparameterized_input)
        return torch.where(
            corrupted_input == 1.0,
            self._nearly_one,
            torch.where(
                corrupted_input == 0.0, self._nearly_zero, decoded_input
            ),
        )
