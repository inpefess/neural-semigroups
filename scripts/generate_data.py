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
from tqdm import tqdm

from neural_semigroups.magma import Magma
from neural_semigroups.utils import random_magma


def generate_data():
    """ main function """
    cardinality = 3
    total = 1000000
    positive = dict()
    negative = dict()
    for _ in tqdm(range(total)):
        cayley_table = random_magma(cardinality)
        if Magma(cayley_table).is_associative:
            positive[str(cayley_table)] = cayley_table
        elif len(positive) > len(negative):
            negative[str(cayley_table)] = cayley_table
    np.savez(
        f"databases/semigroup.{cardinality}.npz",
        database=np.stack(list(positive.values()) + list(negative.values())),
        labels=np.concatenate([
            np.ones(len(positive), dtype=np.int64),
            np.zeros(len(negative), dtype=np.int64)
        ])
    )


if __name__ == "__main__":
    generate_data()
