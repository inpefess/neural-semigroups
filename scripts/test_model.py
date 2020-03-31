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
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
from tqdm import tqdm

from neural_semigroups.cayley_database import CayleyDatabase
from neural_semigroups.denoising_autoencoder import MagmaDAE


def get_test_arguments() -> Namespace:
    """
    get script arguments

    :returns: script parameters
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--cardinality",
        type=int,
        help="semigroup cardinality",
        required=True,
        choices=range(2, 8)
    )
    return parser.parse_args()


def load_pre_trained_model(cardinality: int) -> CayleyDatabase:
    """
    load pre-trained model and database of Cayley tables

    :param cardinality: semigroup cardinality
    :returns: a semigroups searching object
    """
    cayley_db = CayleyDatabase(cardinality)
    cayley_db.load_model(f"semigroups.{cardinality}.model")
    if isinstance(cayley_db.model, MagmaDAE):
        cayley_db.model.apply_corruption = False
    return cayley_db


def main():
    """ build and show a pre-trained model quality report """
    cardinality = get_test_arguments().cardinality
    cayley_db = load_pre_trained_model(cardinality)
    max_level = cardinality ** 2 // 2
    totals = np.zeros((3, max_level), dtype=np.int32)
    database_size = len(cayley_db.database)
    test_indices = np.random.choice(
        range(database_size),
        min(database_size, 1000),
        replace=False
    )
    for i in tqdm(test_indices):
        cayley_table = cayley_db.database[i]
        for level in range(1, max_level + 1):
            rows, cols = zip(*[
                (point // cardinality, point % cardinality)
                for point in np.random.randint(0, cardinality ** 2, level)
            ])
            puzzle = cayley_table.copy()
            puzzle[rows, cols] = -1
            solution, _ = cayley_db.fill_in_with_model(puzzle)
            totals[0, level - 1] += 1
            guessed_cells = sum(
                solution[rows, cols] == cayley_table[rows, cols]
            )
            if guessed_cells == level:
                totals[1, level - 1] += 1
            totals[2, level - 1] += guessed_cells
    print_report(totals)


def print_report(totals: np.ndarray) -> None:
    """
    print report in a pretty format

    :param totals: a table with three columns:
    * a column with total number of puzzles per level
    * a column with numbers of correctly solved puzzles
    * numbers of correctly guessed cells in all puzzles
    :returns:
    """
    levels = range(1, totals.shape[1] + 1)
    hidden_cells = totals[0] * levels
    report = pd.DataFrame({
        "hidden cells per puzzle": levels,
        "puzzles": totals[0],
        "solved puzzles": totals[1],
        "solved puzzles (%)": totals[1] * 100 // totals[0],
        "hidden cells": hidden_cells,
        "guessed cells": totals[2],
        "guessed cells (%)": totals[2] * 100 // hidden_cells
    })
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
