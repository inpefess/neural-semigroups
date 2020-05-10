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
from collections import namedtuple
from datetime import datetime
from typing import Dict, Tuple, Union

import torch
from ignite.contrib.handlers.tensorboard_logger import (
    OutputHandler,
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Engine,
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

from neural_semigroups.associator_loss import AssociatorLoss
from neural_semigroups.cayley_database import CayleyDatabase
from neural_semigroups.constants import CURRENT_DEVICE
from neural_semigroups.magma import Magma
from neural_semigroups.utils import get_newest_file


def load_database_as_cubes(
    cardinality: int, train_size: int, validation_size: int
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
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
        torch.stack(train_cubes),
        torch.stack(validation_cubes),
        torch.stack(test_cubes),
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


# pylint: disable=unused-argument
def associative_ratio(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    a wrapper around a discrete case of ``AssociatorLoss``

    :param prediction: a batch of generated Cayley tables
    :param target: unused argument needed for compatibility
    :returns: a percentage of associative tables in a batch
    """
    return AssociatorLoss(True)(prediction)


def get_associator_evaluator(model: Module, loss: Module) -> Engine:
    """
    get an ``ignite`` evaluator for semigroups completion task

    :param model: a network to train
    :param loss: a loss to minimise (usually based on an ``AssociatorLoss``)
    :returns: an ``ignite`` evaluator
    """
    return create_supervised_evaluator(
        model,
        {"loss": Loss(loss), "associative_ratio": Loss(associative_ratio)},
        CURRENT_DEVICE,
    )


ThreeEvaluators = namedtuple(
    "ThreeEvaluators", ["train", "validation", "test"]
)


def get_three_evaluators(model: Module, loss: Module) -> ThreeEvaluators:
    """
    a factory of named tuples ``ThreeEvaluators``

    >>> isinstance(get_three_evaluators(Module(), Module()), ThreeEvaluators)
    True

    :param model: a network to train
    :param loss: a loss to minimise during training
    :returns: a triple of train, validation, and test ``ignite`` evaluators
    """
    return ThreeEvaluators(
        train=get_associator_evaluator(model, loss),
        validation=get_associator_evaluator(model, loss),
        test=get_associator_evaluator(model, loss),
    )


def add_early_stopping_and_checkpoint(
    evaluator: Engine, trainer: Engine, checkpoint_filename: str, model: Module
) -> None:
    """
    adds two event handlers to an ``ignite`` trainer/evaluator pair:

    * early stopping
    * best model checkpoint saver

    :param evaluator: an evaluator to add hooks to
    :param trainer: a trainer from which to make a checkpoint
    :param checkpoint_filename: some pretty name for a checkpoint
    :param model: a network which is saved in checkpoints
    """

    def score(engine):
        return -engine.state.metrics["loss"]

    early_stopping = EarlyStopping(10, score, trainer)
    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    checkpoint = ModelCheckpoint(
        "checkpoints", "", score_function=score, require_empty=False
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint, {checkpoint_filename: model}
    )


def learning_pipeline(
    params: Dict[str, Union[int, float]],
    cardinality: int,
    model: Module,
    loss: Loss,
    data_loaders: Tuple[DataLoader, DataLoader, DataLoader],
) -> None:
    """
    run a comon learning pipeline

    :param params: parameters of learning: epochs, learning_rate.
    :param cardinality: a semigroup cardinality
    :param model: a network architecture
    :param loss: the criterion to optimize
    :param data_loaders: train, validation, and test data loaders
    """
    trainer = create_supervised_trainer(
        model,
        Adam(model.parameters(), lr=params["learning_rate"]),
        loss,
        CURRENT_DEVICE,
    )
    evaluators = get_three_evaluators(model, loss)

    @trainer.on(Events.EPOCH_COMPLETED)
    # pylint: disable=unused-argument,unused-variable
    def validate(trainer):
        evaluators.train.run(data_loaders[0])
        evaluators.validation.run(data_loaders[1])

    @trainer.on(Events.COMPLETED)
    # pylint: disable=unused-argument,unused-variable
    def test(trainer):
        model.load_state_dict(torch.load(get_newest_file("checkpoints")))
        evaluators.test.run(data_loaders[2])

    add_early_stopping_and_checkpoint(
        evaluators.validation, trainer, f"semigroup{cardinality}", model
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
    tb_logger.attach(evaluators.train, training_loss, Events.COMPLETED)
    validation_loss = OutputHandler(
        "validation",
        ["loss", "associative_ratio"],
        global_step_transform=global_step_from_engine(trainer),
    )
    tb_logger.attach(evaluators.validation, validation_loss, Events.COMPLETED)
    test_loss = OutputHandler(
        "test",
        ["loss", "associative_ratio"],
        global_step_transform=global_step_from_engine(trainer),
    )
    tb_logger.attach(evaluators.test, test_loss, Events.COMPLETED)
    trainer.run(data_loaders[0], max_epochs=params["epochs"])
    tb_logger.close()


def get_loaders(
    cardinality: int,
    batch_size: int,
    train_size: int,
    validation_size: int,
    use_labels: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    get train and validation data loaders

    :param cardinality: the cardinality of a ``smallsemi`` database
    :param batch_size: batch size (common for train and validation)
    :param train_size: number of tables for training
    :param validation_size: number of tables for validation
    :param use_labels: whether to set a target as labels from database (for
    classifier) or to use X's as labels (for autoencoder)
    :returns: a triple of train, validation, and test data loaders
    """
    (
        train_tensor,
        val_tensor,
        test_tensor,
        train_labels,
        validation_labels,
        test_labels,
    ) = load_database_as_cubes(cardinality, train_size, validation_size)
    if use_labels:
        train_data = TensorDataset(train_tensor, train_labels)
        val_data = TensorDataset(val_tensor, validation_labels)
        test_data = TensorDataset(test_tensor, test_labels)
    else:
        train_data = TensorDataset(train_tensor, train_tensor)
        val_data = TensorDataset(val_tensor, val_tensor)
        test_data = TensorDataset(test_tensor, test_tensor)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(val_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size, shuffle=True),
    )
