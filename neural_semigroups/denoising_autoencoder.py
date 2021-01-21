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
from collections import OrderedDict
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential, Softmax2d

from neural_semigroups.constants import CURRENT_DEVICE


def get_linear_bn_relu_sequence(
    dims: List[int], layer_name_postfix: str
) -> Module:
    """
    constructs a sequential model of triples of
    linear, relu, and batch norm layers of given dimensions

    :param dims: dimensions for layers in a sequence
    :param layer_name_postfix:
    :returns: a sequential module
    """
    layers: "OrderedDict[str, Module]" = OrderedDict()
    for i in range(len(dims) - 1):
        layers.update(
            {
                f"linear{layer_name_postfix}{i}": Linear(
                    dims[i], dims[i + 1], bias=True
                )
            }
        )
        layers.update({f"relu{layer_name_postfix}{i}": ReLU()})
        layers.update({f"bn{layer_name_postfix}{i}": BatchNorm1d(dims[i + 1])})
    return Sequential(layers).to(CURRENT_DEVICE)


def get_encoder_and_decoder_layers(
    dims: List[int], split_last: bool
) -> Tuple[Module, Module]:
    """
    construct symmetrical encoder and decoder modules

    :param dims: the dimensions of layers in an encoder part (the same dimensions are used for a decoder)
    :param split_last: if ``True``, makes their last encoder dimensions twice larger than the first decoder dimension (for a reparametrization trick)
    :returns: a pair of two sequential models, representing encoder and decoder layers
    """
    encoder_dims = dims.copy()
    decoder_dims = list(reversed(dims))
    if split_last:
        encoder_dims[-1] *= 2
    encoder_layers = get_linear_bn_relu_sequence(encoder_dims, "0")
    decoder_layers = get_linear_bn_relu_sequence(decoder_dims, "1")
    return encoder_layers, decoder_layers


# pylint: disable=abstract-method
class MagmaDAE(Module):
    """
    Denoising Autoencoder for probability Cayley cubes of magmas
    """

    def __init__(
        self,
        cardinality: int,
        hidden_dims: List[int],
        do_reparametrization: bool = False,
    ):
        """
        :param cardinality: the number of elements in a magma
        :param hidden_dims: a list of sizes of hidden layers of the encoder and the decoder
        :param do_reparametrization: if ``True``, adds a reparametrization trick
        """
        super().__init__()
        self.cardinality = cardinality
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

    def reparametrize(self, mu_and_sigma: Tensor) -> Tensor:
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
    def forward(self, cayley_cubes: Tensor) -> Tensor:
        """
        forward pass inherited from Module

        :param cayley_cubes: a batch of probabilistic representations of magmas
        :returns: auto-encoded probabilistic representations of magmas
        """
        encoded_input = self.encode(cayley_cubes)
        reparametrized_input = self.reparametrize(encoded_input)
        decoded_input = self.decode(reparametrized_input)
        return torch.where(
            torch.eq(cayley_cubes, 1.0),
            self._nearly_one,
            torch.where(
                torch.eq(cayley_cubes, 0.0), self._nearly_zero, decoded_input,
            ),
        )
