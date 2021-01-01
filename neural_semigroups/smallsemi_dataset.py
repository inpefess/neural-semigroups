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
import gzip
import tarfile
from glob import glob
from os import path
from typing import Callable, Optional

import requests
from torch.utils.data import TensorDataset

from neural_semigroups.utils import (
    download_file_from_url,
    find_substring_by_pattern,
    import_smallsemi_format,
)


class Smallsemi(TensorDataset):
    """
    a ``torch.util.data.Dataset`` wrapper for the data
    from https://www.gap-system.org/Packages/smallsemi.html
    """

    gap_packages_url = (
        "https://www.gap-system.org/pub/gap/gap4/tar.bz2/packages/"
    )

    def __init__(
        self,
        root: str,
        cardinality: int,
        download: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        :param root: root directory of dataset where
            ``smallsemi-*/data/data2to7/`` exist.
        :param cardinality: a semigroup cardinality to use.
            Corresponds to ``data{cardinality}.gl.gz``.
        :param download: if true, downloads the dataset from the internet
            and puts it in root directory. If dataset is already downloaded,
            it is not downloaded again.
        :param transform: a function/transform that takes in an PIL image
            and returns a transformed version.
        """
        super().__init__()
        self.root = root
        self.cardinality = cardinality
        self.transform = transform
        if download:
            self.download()
        self.load_data_and_labels_from_smallsemi()

    def download(self) -> None:
        """ downloads, unzips and moves ``smallsemi`` data """
        if glob(f"{self.root}/smallsemi-*/data/data2to7/") == []:
            full_name_with_version = find_substring_by_pattern(
                strings=requests.get(self.gap_packages_url).text.split("\n"),
                starts_with="smallsemi",
                ends_before=".tar.bz2",
            )
            archive_path = path.join(
                self.root, f"{full_name_with_version}.tar.bz2"
            )
            download_file_from_url(
                url=f"{self.gap_packages_url}{full_name_with_version}.tar.bz2",
                filename=archive_path,
            )
            with tarfile.open(archive_path) as archive:
                archive.extractall(self.root)

    def load_data_and_labels_from_smallsemi(self) -> None:
        """ loads data from ``smallsemi`` package """
        filenames = glob(
            f"{self.root}/smallsemi-*/"
            + f"data/data2to7/data{self.cardinality}.gl.gz"
        )
        if len(filenames) != 1:
            raise ValueError(
                f"{self.root} must have only one version of smallsemi package"
            )
        with gzip.open(filenames[0], "rb") as file:
            database = import_smallsemi_format(file.readlines())
            self.tensors = (database, database)

    def __getitem__(self, index):
        tensors = super().__getitem__(index)
        if self.transform is not None:
            tensors = self.transform(tensors)
        return tensors