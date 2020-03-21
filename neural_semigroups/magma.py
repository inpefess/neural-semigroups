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
import numpy as np


class Magma:
    """
    an implementation of `a magma`_ (a set with a binary operation)

    .. _a magma: https://en.wikipedia.org/wiki/Magma_%28algebra%29
    """
    cayley_table: np.ndarray

    def __init__(self, cayley_table: np.ndarray):
        """
        :param cayley_table: a Cayley table for a magma
        """
        self.cayley_table = cayley_table

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
                        self.cayley_table[
                            self.cayley_table[one, two], a_range
                        ] !=
                        self.cayley_table[
                            one, self.cayley_table[two, a_range]
                        ]
                ).any():
                    return False
        return True

    @property
    def is_commutative(self) -> bool:
        """
        check commutativity of a Cayley table

        :returns: whether the input table is commutative or not

        """
        return np.allclose(
            self.cayley_table, self.cayley_table.T
        )

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
            if (
                    not np.allclose(
                        np.sort(self.cayley_table[i, :]), identity_row
                    ) or
                    not np.allclose(
                        np.sort(self.cayley_table[:, i]), identity_row
                    )
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
            dtype=np.float32
        )
        for i in range(self.cardinality):
            for j in range(self.cardinality):
                cube[i, j, self.cayley_table[i, j]] = 1.0
        return cube
