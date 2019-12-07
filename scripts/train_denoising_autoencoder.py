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
import logging
from argparse import ArgumentParser, Namespace
from time import time
from typing import Tuple

import numpy as np
import torch
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import EarlyStopping
from ignite.metrics.loss import Loss
from torch import Tensor
from torch.nn.functional import kl_div
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from neural_semigroups.denoising_autoencoder import MagmaDAE
from neural_semigroups.magma import Magma
from neural_semigroups.table_guess import TableGuess, train_test_split


def load_database_as_cubes(
        database_filename: str,
        train_size: int,
        validation_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    load a database file to probability cubes representation

    :param database_filename: the name of the file from which to extract data
    :param train_size: number of tables for training
    :param validation_size: number of tables for validation
    :returns: tree arrays of probability Cayley cubes: train, validation and
    test
    """
    table_guess = TableGuess()
    logging.info("reading data from disk")
    table_guess.load_smallsemi_database(database_filename)
    logging.info("splitting by train and test")
    train, validation, test = train_test_split(
        table_guess, train_size, validation_size
    )
    logging.info("augmenting train set")
    train.augment_by_equivalent_tables()
    logging.info("generating train cubes")
    train_cubes = list()
    for cayley_table in tqdm(train.database):
        train_cubes.append(Magma(cayley_table).probabilistic_cube)
    validation_cubes = list()
    logging.info("generating validation cubes")
    for cayley_table in tqdm(validation.database):
        validation_cubes.append(Magma(cayley_table).probabilistic_cube)
    test_cubes = list()
    logging.info("generating test cubes")
    for cayley_table in tqdm(test.database):
        test_cubes.append(Magma(cayley_table).probabilistic_cube)
    return (
        np.stack(train_cubes), np.stack(validation_cubes), np.stack(test_cubes)
    )


def get_loaders(
        database_filename: str,
        batch_size: int,
        train_size: int,
        validation_size: int
) -> Tuple[DataLoader, DataLoader]:
    """
    get train and validation data loaders

    :param database_filename: the name of the file from which to extract data
    :param batch_size: batch size (common for train and validation)
    :param train_size: number of tables for training
    :param validation_size: number of tables for validation
    :returns: a pair of train and validation data loaders
    """
    train, validation, test = load_database_as_cubes(
        database_filename, train_size, validation_size
    )
    train_tensor = torch.from_numpy(train)
    train_data = TensorDataset(train_tensor, train_tensor)
    val_tensor = torch.from_numpy(validation)
    val_data = TensorDataset(val_tensor, val_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def get_arguments() -> Namespace:
    """
    parse script arguments

    :returns: script parameters
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--cardinality",
        type=int,
        help="magma cardinality",
        required=True,
        choices=range(2, 8)
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="number of epochs to train",
        default=100,
        required=False
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="learning rate",
        default=0.001,
        required=False
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size for training",
        default=32,
        required=False
    )
    parser.add_argument(
        "--train_size",
        type=int,
        help="number of tables for training",
        required=True
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        help="number of tables for validation",
        required=True
    )
    return parser.parse_args()


def main():
    """ train and save the model """
    args = get_arguments()
    cardinality = args.cardinality
    train_loader, val_loader = get_loaders(
        database_filename=f"smallsemi/data{cardinality}.gl",
        batch_size=args.batch_size,
        train_size=args.train_size,
        validation_size=args.validation_size
    )
    logging.info("data prepared")
    model = MagmaDAE(
        cardinality=cardinality,
        hidden_dims=[
            cardinality ** 2,
            cardinality
        ],
        corruption_rate=0.5
    )
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    def loss(prediction: Tensor, target: Tensor) -> Tensor:
        return kl_div(prediction, target, reduce="batchmean")
    trainer = create_supervised_trainer(model, optimizer, loss)
    evaluator = create_supervised_evaluator(
        model,
        metrics={"loss": Loss(loss)}
    )

    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss
    handler = EarlyStopping(
        patience=10, score_function=score_function, trainer=trainer
    )
    evaluator.add_event_handler(Events.COMPLETED, handler)
    writer = SummaryWriter()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        writer.add_scalars(
            "loss",
            {"training": evaluator.state.metrics["loss"]},
            global_step=trainer.state.iteration,
            walltime=int(time())
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        writer.add_scalars(
            "loss",
            {"validation": evaluator.state.metrics["loss"]},
            global_step=trainer.state.iteration,
            walltime=int(time())
        )
        print(evaluator.state.metrics["loss"])
        torch.save(model, f"semigroups.{cardinality}.model")
    logging.info("training started")
    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == "__main__":
    main()
