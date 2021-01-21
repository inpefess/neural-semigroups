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
from typing import Optional, Tuple

import torch


def get_two_indices_per_sample(
    batch_size: int, cardinality: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    generates all possible combination of two indices
    for each sample in a batch

    >>> get_two_indices_per_sample(1, 2)
    (tensor([0, 0, 0, 0]), tensor([0, 0, 1, 1]), tensor([0, 1, 0, 1]))

    :param batch_size: number of samples in a batch
    :param cardinality: number of possible values of an index
    :returns: triples (index, index, index)
    """
    a_range = torch.arange(cardinality)
    return (
        torch.arange(batch_size).repeat_interleave(cardinality * cardinality),
        a_range.repeat_interleave(cardinality).repeat(batch_size),
        a_range.repeat(cardinality).repeat(batch_size),
    )


class Magma:
    """
    an implementation of `a magma`_ (a set with a binary operation)

    .. _a magma: https://en.wikipedia.org/wiki/Magma_%28algebra%29
    """

    cayley_table: torch.Tensor

    def __init__(
        self,
        cayley_table: Optional[torch.Tensor] = None,
        cardinality: Optional[int] = None,
    ):
        """
        constucts a new magma

        >>> seed = torch.manual_seed(11)
        >>> Magma(cardinality=2)
        tensor([[1, 1],
                [0, 1]])
        >>> Magma(torch.tensor([[0, 1], [1, 0]]))
        tensor([[0, 1],
                [1, 0]])
        >>> Magma()
        Traceback (most recent call last):
            ...
        ValueError: at least one argument must be given
        >>> Magma(torch. tensor([[0]]), cardinality=2)
        Traceback (most recent call last):
            ...
        ValueError: cayley_table must be a `torch.Tensor` of shape (cardinality, cardinality)

        :param cayley_table: a Cayley table for a magma.
                             If not provided, a random table is generated.
        :param cardinality: a number of elements in a magma to generate a random one
        """
        if cayley_table is None:
            if cardinality is None:
                raise ValueError("at least one argument must be given")
            self.cayley_table = torch.randint(
                low=0,
                high=cardinality,
                size=[cardinality, cardinality],
                dtype=torch.long,
            )
        else:
            if cayley_table.shape[0] == cayley_table.shape[1] and (
                cardinality is None or cayley_table.shape[0] == cardinality
            ):
                self.cayley_table = cayley_table
            else:
                raise ValueError(
                    "cayley_table must be a `torch.Tensor` of shape "
                    "(cardinality, cardinality)"
                )

    def __repr__(self) -> str:
        return str(self.cayley_table)

    @property
    def cardinality(self) -> int:
        """
        number of elements in a magma
        """
        return self.cayley_table.shape[0]

    @property
    def is_associative(self) -> bool:
        """
        check associativity of a Cayley table

        :returns: whether the input table is associative or not

        """
        a_range = torch.arange(self.cardinality)
        square = self.cardinality ** 2
        one = a_range.repeat_interleave(square)
        two = a_range.repeat_interleave(self.cardinality).repeat(
            self.cardinality
        )
        three = a_range.repeat(square)
        return self.cayley_table[self.cayley_table[one, two], three].equal(
            self.cayley_table[one, self.cayley_table[two, three]]
        )

    @property
    def is_commutative(self) -> bool:
        """
        check commutativity of a Cayley table

        :returns: whether the input table is commutative or not

        """
        return torch.allclose(
            self.cayley_table, self.cayley_table.transpose(0, 1)
        )

    @property
    def identity(self) -> int:
        """
        find an identity element in a Cayley table

        :returns: the index of the identity element or -1 if there is no identity
        """
        a_range = torch.arange(self.cardinality)
        left_identity = (
            torch.eq(self.cayley_table, a_range).min(dim=1)[0].max(dim=0)
        )
        right_identity = (
            torch.eq(self.cayley_table.transpose(0, 1), a_range)
            .min(dim=1)[0]
            .max(dim=0)
        )
        if bool(
            left_identity[0]
            and right_identity[0]
            and left_identity[1].equal(right_identity[1])
        ):
            identity = int(left_identity[1])
        else:
            identity = -1
        return identity

    @property
    def has_inverses(self) -> bool:
        """
        check whether there are solutions of equations :math:`ax=b` and :math:`xa=b``
        """
        a_range = torch.arange(self.cardinality)
        return bool(
            torch.eq(self.cayley_table.sort()[0], a_range).min(dim=1)[0].min()
            and torch.eq(self.cayley_table.transpose(0, 1).sort()[0], a_range)
            .min(dim=1)[0]
            .min()
        )

    @property
    def probabilistic_cube(self) -> torch.Tensor:
        """
        a 3d array :math:`a` where :math:`a_{ijk}=P\\left\\{e_ie_j=e_k\\right\\}`

        :returns: a probabilistic cube representation of a Cayley table
        """
        cube = torch.zeros(
            [self.cardinality, self.cardinality, self.cardinality],
            dtype=torch.float32,
        )
        a_range = torch.arange(self.cardinality)
        one = a_range.repeat_interleave(self.cardinality)
        two = a_range.repeat(self.cardinality)
        cube[one, two, self.cayley_table[one, two]] = 1.0
        return cube

    @property
    def next_magma(self) -> "Magma":
        """
        goes to the next magma Cayley table in their lexicographical order

        >>> Magma(torch.tensor([[0, 1], [1, 0]])).next_magma
        tensor([[0, 1],
                [1, 1]])
        >>> Magma(torch.tensor([[0, 1], [1, 1]])).next_magma
        tensor([[1, 0],
                [0, 0]])
        >>> Magma(torch.tensor([[1, 1], [1, 1]])).next_magma
        Traceback (most recent call last):
            ...
        ValueError: there is no next magma!

        :returns: another magma
        """
        next_table = self.cayley_table.clone().detach()
        row, column = self.cardinality - 1, self.cardinality - 1
        while next_table[row, column] >= self.cardinality - 1:
            next_table[row, column] = 0
            row = row + (column - 1) // self.cardinality
            column = (column - 1) % self.cardinality
            if row < 0:
                raise ValueError("there is no next magma!")
        next_table[row, column] += 1
        return Magma(next_table)

    def random_isomorphism(self) -> torch.Tensor:
        """
        get some Cayley table isomorphic to ``self.cayley_table`` form example
        >>> Magma(torch.tensor([[0, 0], [0, 0]])).random_isomorphism()
        tensor([[1, 1],
                        [1, 1]])
        """
        permutation_tensor = torch.randperm(self.cayley_table.shape[0]).to(
            self.cayley_table.device
        )
        _, one, two = get_two_indices_per_sample(1, self.cayley_table.shape[0])
        isomorphic_cayley_table = torch.zeros(
            self.cayley_table.shape,
            dtype=torch.long,
            device=self.cayley_table.device,
        )
        isomorphic_cayley_table[
            permutation_tensor[one], permutation_tensor[two]
        ] = permutation_tensor[self.cayley_table[one, two]]
        return isomorphic_cayley_table
