"""
   Copyright 2019-2022 Boris Shminke

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
from typing import Tuple, Union

import torch
from torch.utils.data import IterableDataset

from neural_semigroups.constants import CURRENT_DEVICE


class RandomDataset(IterableDataset):
    """
    an iterable dataset having fixed length and returning random tensors of
    pre-defined shape

    >>> data = RandomDataset(2, ([5, 2], [1]))
    >>> print([item.shape for item in data[1]])
    [torch.Size([5, 2]), torch.Size([1])]
    >>> for row in data:
    ...     print([item.shape for item in row])
    ...     break
    [torch.Size([5, 2]), torch.Size([1])]
    >>> data = RandomDataset(3, [4, 4])
    >>> print(data[1].shape)
    torch.Size([4, 4])
    >>> for row in data:
    ...     print(row.shape)
    ...     break
    torch.Size([4, 4])
    >>> print(len(data))
    3
    """

    def __init__(
        self,
        data_size: int,
        data_dim: Union[torch.Size, Tuple[torch.Size, ...]],
    ):
        super().__init__()
        self.data_size = data_size
        self.data_dim = data_dim

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        if isinstance(self.data_dim, tuple):
            return tuple(
                torch.rand(dim).to(CURRENT_DEVICE) for dim in self.data_dim
            )
        return torch.rand(self.data_dim).to(CURRENT_DEVICE)

    def __iter__(self):
        if isinstance(self.data_dim, Tuple):
            return iter(
                tuple(
                    torch.rand(dim).to(CURRENT_DEVICE) for dim in self.data_dim
                )
                for i in range(self.data_size)
            )
        return iter(
            torch.rand(self.data_dim).to(CURRENT_DEVICE)
            for i in range(self.data_size)
        )
