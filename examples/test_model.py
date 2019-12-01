"""
   Copyright 2019 Boris Shminke

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

from neural_semigroups.magma import Magma
from neural_semigroups.table_guess import TableGuess


def get_arguments() -> Namespace:
    """
    parse script arguments

    :returns: script parameters
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--cardinality",
        type=int,
        help="semigroup cardinality",
        required=True
    )
    return parser.parse_args()


def load_pre_trained_model(cardinality: int) -> TableGuess:
    """
    load pre-trained model and database of Cayley tables

    :param cardinality: semigroup cardinality
    :returns: a semigroups searching object
    """
    table_guess = TableGuess()
    table_guess.load_smallsemi_database(f"smallsemi/data{cardinality}.gl")
    table_guess.load_model(f"semigroups.{cardinality}.model")
    table_guess.model.apply_corruption = False
    return table_guess


def main():
    """ build and show a pre-trained model quality report """
    cardinality = get_arguments().cardinality
    table_guess = load_pre_trained_model(cardinality)
    max_level = cardinality ** 2
    total_tables = np.zeros(max_level, dtype=np.int32)
    total_cells = np.zeros(max_level, dtype=np.int32)
    correct_tables = np.zeros(max_level, dtype=np.int32)
    correct_cells = np.zeros(max_level, dtype=np.int32)
    for cayley_table in tqdm(table_guess.database):
        for level in range(1, max_level + 1):
            rows = list()
            cols = list()
            for point in np.random.randint(0, cardinality ** 2, level):
                rows.append(point // cardinality)
                cols.append(point % cardinality)
            puzzle = cayley_table.copy()
            puzzle[rows, cols] = -1
            solution, _ = table_guess.predict_from_model(puzzle)
            total_tables[level - 1] += 1
            total_cells[level - 1] += level
            if Magma(solution).is_associative:
                correct_tables[level - 1] += 1
                correct_cells[level - 1] += level
            else:
                correct_cells[level - 1] += sum(
                    solution[rows, cols] == cayley_table[rows, cols]
                )
    print_report(total_tables, correct_tables, total_cells, correct_cells)


def print_report(
        total_tables: np.ndarray,
        correct_tables: np.ndarray,
        total_cells: np.ndarray,
        correct_cells: np.ndarray
) -> None:
    """
    print report in a pretty format

    :param total_tables: a column with total number of puzzles per level
    :param correct_tables: a column with numbers of correctly solved puzzles
    :param total_cells: total numbers of hidden cells in all puzzles
    :param correct_cells: numbers of correctly guessed cells in all puzzles
    :returns:
    """
    report = pd.DataFrame({
        "hidden cells per puzzle": range(1, total_tables.shape[0] + 1),
        "puzzles": total_tables,
        "solved puzzles": correct_tables,
        "solved puzzles (%)": correct_tables * 100 // total_tables,
        "hidden cells": total_cells,
        "guessed cells": correct_cells,
        "guessed cells (%)": correct_cells * 100 // total_cells
    })
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
