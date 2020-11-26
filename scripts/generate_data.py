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
import torch
from tqdm import tqdm

from neural_semigroups.magma import Magma


def generate_data():
    """ main function """
    cardinality = 3
    positive = dict()
    negative = dict()
    for _ in tqdm(range(1000000)):
        magma = Magma(cardinality=cardinality)
        if magma.is_associative:
            positive[magma] = magma.cayley_table
        elif len(positive) > len(negative):
            negative[magma] = magma.cayley_table
    torch.save(
        {
            "database": torch.stack(
                list(positive.values()) + list(negative.values())
            ),
            "labels": torch.cat(
                [
                    torch.ones(len(positive), dtype=torch.long),
                    torch.zeros(len(negative), dtype=torch.long),
                ]
            ),
        },
        f"databases/semigroup.{cardinality}.zip",
        _use_new_zipfile_serialization=True,
    )


if __name__ == "__main__":
    generate_data()
