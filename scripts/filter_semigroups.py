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
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from neural_semigroups.magma import Magma


def main():
    """ do all """
    parser = ArgumentParser()
    parser.add_argument("--dim", type=int, help="magma cardinality")
    choices = ["identity", "inverses"]
    parser.add_argument(
        "--filter", choices=choices, help="additional property"
    )
    args = parser.parse_args()
    input_file = "semigroup" if args.filter == choices[0] else "monoid"
    with torch.load(
        f"./databases/{input_file}.{args.dim}.zip"
    ) as torch_zip_file:
        cayley_tables = torch_zip_file["database"]
    output_file = "monoid" if args.filter == choices[0] else "group"
    filtered = list()
    for cayley_table in tqdm(cayley_tables):
        append_to_list = False
        if args.filter == choices[0]:
            append_to_list = Magma(cayley_table).identity >= 0
        elif args.filter == choices[1]:
            append_to_list = Magma(cayley_table).has_inverses
        if append_to_list:
            filtered.append(cayley_table)
    torch.save(
        {"database": torch.stack(filtered)},
        f"./databases/{output_file}.{args.dim}.zip",
        _use_new_zipfile_serialization=True,
    )


if __name__ == "__main__":
    main()
