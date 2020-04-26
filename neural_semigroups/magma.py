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
from typing import Optional

import numpy as np


class Magma:
    """
    an implementation of `a magma`_ (a set with a binary operation)

    .. _a magma: https://en.wikipedia.org/wiki/Magma_%28algebra%29
    """

    cayley_table: np.ndarray

    def __init__(
        self,
        cayley_table: Optional[np.ndarray] = None,
        cardinality: Optional[int] = None,
    ):
        """
        constucts a new magma

        >>> np.random.seed(777)
        >>> Magma(cardinality=2)
        [[1 1]
         [1 0]]
        >>> Magma(np.array([[0, 1], [1, 0]]))
        [[0 1]
         [1 0]]
        >>> Magma()
        Traceback (most recent call last):
            ...
        ValueError: at least one argument must be given
        >>> Magma(np.array([[0]]), 2)
        Traceback (most recent call last):
            ...
        ValueError: inconsistent argument values
        >>> Magma([[0, 1]])
        Traceback (most recent call last):
            ...
        ValueError: a cayley_table must be a numpy array of shape (n, n)

        :param cayley_table: a Cayley table for a magma.
                             If not provided, a random table is generated.
        :param cardinality: a number of elements in a magma to generate a random one
        """
        if cayley_table is None:
            if cardinality is None:
                raise ValueError("at least one argument must be given")
            self.cayley_table = np.random.randint(
                low=0, high=cardinality, size=cardinality * cardinality
            ).reshape(cardinality, cardinality)
        else:
            all_right = False
            if isinstance(cayley_table, np.ndarray):
                if len(cayley_table.shape) == 2:
                    if cayley_table.shape[0] == cayley_table.shape[1]:
                        self.cayley_table = cayley_table
                        all_right = True
            if cardinality is not None and all_right:
                if cayley_table.shape[0] != cardinality:
                    raise ValueError("inconsistent argument values")
            if not all_right:
                raise ValueError(
                    "a cayley_table must be a numpy array of shape (n, n)"
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

        :returns: whether the input table is assosiative or not

        """
        a_range = range(self.cardinality)
        for one in a_range:
            for two in a_range:
                if (
                    self.cayley_table[self.cayley_table[one, two], a_range]
                    != self.cayley_table[one, self.cayley_table[two, a_range]]
                ).any():
                    return False
        return True

    @property
    def is_commutative(self) -> bool:
        """
        check commutativity of a Cayley table

        :returns: whether the input table is commutative or not

        """
        return np.allclose(self.cayley_table, self.cayley_table.T)

    @property
    def identity(self) -> int:
        """
        find an identity element in a Cayley table

        :returns: the index of the identity element or -1 if there is no identity
        """
        identity_row = np.arange(self.cardinality)
        identity = -1
        for i in identity_row:
            if np.allclose(self.cayley_table[i, :], identity_row):
                identity = i
                break
        return identity

    @property
    def has_inverses(self) -> bool:
        """
        check whether there are solutions of equations :math:`ax=b` and :math:`xa=b``
        """
        identity_row = np.arange(self.cardinality)
        has_inverses = True
        for i in identity_row:
            if not np.allclose(
                np.sort(self.cayley_table[i, :]), identity_row
            ) or not np.allclose(
                np.sort(self.cayley_table[:, i]), identity_row
            ):
                has_inverses = False
                break
        return has_inverses

    @property
    def probabilistic_cube(self) -> np.ndarray:
        """
        a 3d array :math:`a` where :math:`a_{ijk}=P\\left\\{e_ie_j=e_k\\right\\}`

        :returns: a probabilistic cube representation of a Cayley table
        """
        cube = np.zeros(
            [self.cardinality, self.cardinality, self.cardinality],
            dtype=np.float32,
        )
        for i in range(self.cardinality):
            for j in range(self.cardinality):
                cube[i, j, self.cayley_table[i, j]] = 1.0
        return cube

    @property
    def next_magma(self) -> "Magma":
        """
        goes to the next magma Cayley table in their lexicographical order

        >>> Magma(np.array([[0, 1], [1, 0]])).next_magma
        [[0 1]
        [1 1]]
        >>> Magma(np.array([[0, 1], [1, 1]])).next_magma
        [[1 0]
        [0 0]]
        >>> Magma(np.array([[1, 1], [1, 1]])).next_magma
        Traceback (most recent call last):
            ...
        ValueError: there is no next magma!

        :returns: another magma
        """
        next_table = self.cayley_table.copy()
        one = 1
        row = self.cardinality - 1
        column = self.cardinality - 1
        while one == 1:
            if next_table[row, column] < self.cardinality - 1:
                next_table[row, column] += 1
                one = 0
            else:
                if column > 0:
                    next_table[row, column] = 0
                    column -= 1
                else:
                    if row > 0:
                        next_table[row, column] = 0
                        row -= 1
                        column = self.cardinality - 1
                    else:
                        raise ValueError("there is no next magma!")
        return Magma(next_table)
