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
from datetime import datetime
from typing import Dict, Tuple, Union

import numpy as np
import torch
from ignite.contrib.handlers.tensorboard_logger import (
    OutputHandler,
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics.loss import Loss
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from neural_semigroups.cayley_database import CayleyDatabase
from neural_semigroups.constants import CURRENT_DEVICE
from neural_semigroups.magma import Magma


def load_database_as_cubes(
    cardinality: int, train_size: int, validation_size: int
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    load a database file to probability cubes representation

    :param cardinality: cardinality of Cayley database (from ``smallsemi``)
    :param train_size: number of tables for training
    :param validation_size: number of tables for validation
    :returns: three arrays of probability Cayley cubes (train, validation, test
    ) and three arrays of labels for them
    """
    cayley_db = CayleyDatabase(cardinality)
    train, validation, test = cayley_db.train_test_split(
        train_size, validation_size
    )
    train.augment_by_equivalent_tables()
    train_cubes = list()
    for cayley_table in tqdm(train.database, desc="generating train cubes"):
        train_cubes.append(Magma(cayley_table).probabilistic_cube)
    validation_cubes = list()
    for cayley_table in tqdm(
        validation.database, desc="generating validation cubes"
    ):
        validation_cubes.append(Magma(cayley_table).probabilistic_cube)
    test_cubes = list()
    for cayley_table in tqdm(test.database, desc="generating test cubes"):
        test_cubes.append(Magma(cayley_table).probabilistic_cube)
    return (
        np.stack(train_cubes),
        np.stack(validation_cubes),
        np.stack(test_cubes),
        train.labels,
        validation.labels,
        test.labels,
    )


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
        choices=range(2, 8),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="number of epochs to train",
        default=100,
        required=False,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="learning rate",
        default=0.001,
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size for training",
        default=32,
        required=False,
    )
    parser.add_argument(
        "--train_size",
        type=int,
        help="number of tables for training",
        required=True,
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        help="number of tables for validation",
        required=True,
    )
    return parser.parse_args()


def learning_pipeline(
    params: Dict[str, Union[int, float]],
    cardinality: int,
    model: Module,
    loss: Loss,
    data_loaders: Tuple[DataLoader, DataLoader],
) -> None:
    """
    run a comon learning pipeline

    :param params: parameters of learning: epochs, learning_rate.
    :param cardinality: a semigroup cardinality
    :param model: a network architecture
    :param loss: the criterion to optimize
    :oaram data_loader: train and validation data loaders
    """
    trainer = create_supervised_trainer(
        model, Adam(model.parameters(), lr=params["learning_rate"]), loss
    )
    train_evaluator = create_supervised_evaluator(model, {"loss": Loss(loss)})
    evaluator = create_supervised_evaluator(model, {"loss": Loss(loss)})

    @trainer.on(Events.EPOCH_COMPLETED)
    # pylint: disable=unused-argument,unused-variable
    def validate(trainer):
        train_evaluator.run(data_loaders[0])
        evaluator.run(data_loaders[1])

    def score(engine):
        return -engine.state.metrics["loss"]

    early_stopping = EarlyStopping(10, score, trainer)
    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    checkpoint = ModelCheckpoint(
        "checkpoints", "", score_function=score, require_empty=False
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint, {f"semigroup{cardinality}": model}
    )
    ProgressBar().attach(
        trainer,
        output_transform=lambda x: x,
        event_name=Events.EPOCH_COMPLETED,
        closing_event_name=Events.COMPLETED,
    )
    tb_logger = TensorboardLogger(
        log_dir=f"runs/{datetime.now()}", flush_secs=1
    )
    training_loss = OutputHandler(
        "training",
        ["loss"],
        global_step_transform=global_step_from_engine(trainer),
    )
    tb_logger.attach(train_evaluator, training_loss, Events.COMPLETED)
    validation_loss = OutputHandler(
        "validation",
        ["loss"],
        global_step_transform=global_step_from_engine(trainer),
    )
    tb_logger.attach(evaluator, validation_loss, Events.COMPLETED)
    trainer.run(data_loaders[0], max_epochs=params["epochs"])
    tb_logger.close()


def get_loaders(
    cardinality: int,
    batch_size: int,
    train_size: int,
    validation_size: int,
    use_labels: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    get train and validation data loaders

    :param cardinality: the cardinality of a ``smallsemi`` database
    :param batch_size: batch size (common for train and validation)
    :param train_size: number of tables for training
    :param validation_size: number of tables for validation
    :param use_labels: whether to set a target as labels from database (for
    classifier) or to use X's as labels (for autoencoder)
    :returns: a pair of train and validation data loaders
    """
    (
        train,
        validation,
        _,
        train_labels,
        validation_labels,
        _,
    ) = load_database_as_cubes(cardinality, train_size, validation_size)
    train_tensor = torch.from_numpy(train).to(CURRENT_DEVICE)
    val_tensor = torch.from_numpy(validation).to(CURRENT_DEVICE)
    if use_labels:
        train_data = TensorDataset(
            train_tensor, torch.from_numpy(train_labels).to(CURRENT_DEVICE)
        )
        val_data = TensorDataset(
            val_tensor, torch.from_numpy(validation_labels).to(CURRENT_DEVICE)
        )
    else:
        train_data = TensorDataset(train_tensor, train_tensor)
        val_data = TensorDataset(val_tensor, val_tensor)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(val_data, batch_size=batch_size, shuffle=True),
    )
