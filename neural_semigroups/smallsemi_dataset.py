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
import os
import tarfile
from glob import glob
from typing import Callable, Optional

import requests

from neural_semigroups.semigroups_dataset import SemigroupsDataset
from neural_semigroups.utils import (
    download_file_from_url,
    find_substring_by_pattern,
    gunzip,
    import_smallsemi_format,
)


class Smallsemi(SemigroupsDataset):
    """
    a ``torch.util.data.Dataset`` wrapper for the data
    from https://www.gap-system.org/Packages/smallsemi.html

    >>> import shutil
    >>> from neural_semigroups.constants import TEST_TEMP_DATA
    >>> shutil.rmtree(TEST_TEMP_DATA, ignore_errors=True)
    >>> os.mkdir(TEST_TEMP_DATA)
    >>> smallsemi = Smallsemi(root=TEST_TEMP_DATA, cardinality=2)
    Traceback (most recent call last):
       ...
    ValueError: test_temp_data must have exactly one version of smallsemi
    >>> smallsemi = Smallsemi(
    ...     root=TEST_TEMP_DATA,
    ...     cardinality=2,
    ...     download=True,
    ...     transform=lambda x: x
    ... )
    >>> smallsemi[0][0]
    tensor([[0, 0],
            [0, 0]])
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
        :param transform: a function/transform that takes in a Cayley table
            and returns a transformed version.
        """
        super().__init__(root, cardinality, transform)
        if download:
            self.download()
        self.load_data_and_labels_from_smallsemi()

    def download(self) -> None:
        """downloads, unzips and moves ``smallsemi`` data"""
        data_paths = f"{self.root}/smallsemi-*/data/data2to7/data*.gl.gz"
        if not glob(data_paths):
            full_name_with_version = find_substring_by_pattern(
                strings=requests.get(self.gap_packages_url).text.split("\n"),
                starts_with="smallsemi",
                ends_before=".tar.bz2",
            )
            archive_path = os.path.join(
                self.root, f"{full_name_with_version}.tar.bz2"
            )
            download_file_from_url(
                url=f"{self.gap_packages_url}{full_name_with_version}.tar.bz2",
                filename=archive_path,
            )
            with tarfile.open(archive_path) as archive:
                archive.extractall(self.root)
            for archive_path in glob(data_paths):
                gunzip(archive_path)

    def load_data_and_labels_from_smallsemi(self) -> None:
        """loads data from ``smallsemi`` package"""
        filenames = glob(
            f"{self.root}/smallsemi-*/"
            + f"data/data2to7/data{self.cardinality}.gl"
        )
        if len(filenames) != 1:
            raise ValueError(
                f"{self.root} must have exactly one version of smallsemi"
            )
        with open(filenames[0], "r", encoding="utf-8") as file:
            database = import_smallsemi_format(file.readlines())
            self.tensors = (database, database)
