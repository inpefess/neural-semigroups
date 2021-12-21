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
from typing import Callable, Optional

from torch.utils.data import TensorDataset


# pylint: disable=too-few-public-methods
class SemigroupsDataset(TensorDataset):
    """
    an extension of ``torch.util.data.TensorDataset``
    similar to a private class ``torchvision.datasets.vision.VisionDataset``
    """

    def __init__(
        self,
        root: str,
        cardinality: int,
        transform: Optional[Callable] = None,
    ):
        """
        :param root: root directory of dataset
        :param cardinality: a semigroup cardinality to use.
        :param transform: a function/transform that takes in a Cayley table
            and returns a transformed version.
        """
        super().__init__()
        self.root = root
        self.cardinality = cardinality
        self.transform = transform

    def __getitem__(self, index):
        tensors = super().__getitem__(index)
        if self.transform is not None:
            tensors = self.transform(tensors)
        return tensors
