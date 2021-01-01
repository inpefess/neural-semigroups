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
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Union

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
from ignite.metrics import Metric, RunningAverage
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

from neural_semigroups.associator_loss import AssociatorLoss
from neural_semigroups.constants import CURRENT_DEVICE
from neural_semigroups.precise_guess_loss import PreciseGuessLoss
from neural_semigroups.utils import get_newest_file


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


def guessed_ratio(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    a wrapper around a discrete case of ``PreciseGuessLoss``

    :param prediction: a batch of generated Cayley cubes
    :param target: a batch of Cayley cubes from validation
    :returns: a percentage of correctly guessed tables
    """
    return PreciseGuessLoss()(prediction, target)


class ThreeEvaluators:
    """ a triple of three ``ignite`` evaluators: train, validation, test"""

    def __init__(self, model: Module, metrics: Dict[str, Metric]):
        """

        :param model: a network to train
        :param metrics: a dictionary of metrics to evaluate
        :returns:
        """
        self._train = create_supervised_evaluator(
            model, metrics, CURRENT_DEVICE
        )
        self._validation = create_supervised_evaluator(
            model, metrics, CURRENT_DEVICE
        )
        self._test = create_supervised_evaluator(
            model, metrics, CURRENT_DEVICE
        )

    @property
    def train(self):
        """ train evaluator """
        return self._train

    @property
    def validation(self):
        """ validation evaluator """
        return self._validation

    @property
    def test(self):
        """ test evaluator """
        return self._test


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

    early_stopping = EarlyStopping(100, score, trainer)
    evaluator.add_event_handler(Events.COMPLETED, early_stopping)
    checkpoint = ModelCheckpoint(
        "checkpoints", "", score_function=score, require_empty=False
    )
    evaluator.add_event_handler(
        Events.COMPLETED, checkpoint, {checkpoint_filename: model}
    )


def get_tensorboard_logger(
    trainer: Engine, evaluators: ThreeEvaluators, metric_names: List[str]
) -> TensorboardLogger:
    """
    creates a ``tensorboard`` logger which read metrics from given evaluators and attaches it to a given trainer

    :param trainer: an ``ignite`` trainer to attach to
    :param evaluators: a triple of train, validation, and test evaluators to get metrics from
    :param metric_names: a list of metrics to log during validation and testing
    """
    tb_logger = TensorboardLogger(
        log_dir=f"runs/{datetime.now()}", flush_secs=1
    )
    training_loss = OutputHandler(
        "training",
        ["running_loss"],
        global_step_transform=global_step_from_engine(trainer),
    )
    tb_logger.attach(trainer, training_loss, Events.EPOCH_COMPLETED)
    validation_loss = OutputHandler(
        "validation",
        metric_names,
        global_step_transform=global_step_from_engine(trainer),
    )
    tb_logger.attach(evaluators.validation, validation_loss, Events.COMPLETED)
    test_loss = OutputHandler(
        "test",
        metric_names,
        global_step_transform=global_step_from_engine(trainer),
    )
    tb_logger.attach(evaluators.test, test_loss, Events.COMPLETED)
    return tb_logger


def get_trainer(model: Module, learning_rate: float, loss: Callable) -> Engine:
    """
    construct a trainer ``ignite`` engine with pre-attached progress bar and loss running average

    :param model: a network to train
    :param learning_rate: the learning rate for training
    :param loss: a loss to minimise during training
    :returns: an ``ignite`` trainer
    """
    trainer = create_supervised_trainer(
        model,
        Adam(model.parameters(), lr=learning_rate),
        loss,
        CURRENT_DEVICE,
    )
    RunningAverage(output_transform=lambda x: x).attach(
        trainer, "running_loss"
    )
    ProgressBar().attach(
        trainer,
        output_transform=lambda x: x,
        event_name=Events.EPOCH_COMPLETED,
        closing_event_name=Events.COMPLETED,
    )
    return trainer


def learning_pipeline(
    params: Dict[str, Union[int, float]],
    model: Module,
    loss: Callable,
    metrics: Dict[str, Metric],
    data_loaders: Tuple[DataLoader, DataLoader, DataLoader],
) -> None:
    """
    run a common learning pipeline

    :param params: parameters of learning: epochs, learning_rate.
    :param model: a network architecture
    :param loss: the criterion to optimize
    :param metrics: a dictionary of additional metrics to evaluate
    :param data_loaders: train, validation, and test data loaders
    """
    trainer = get_trainer(model, params["learning_rate"], loss)
    evaluators = ThreeEvaluators(model, metrics)

    @trainer.on(Events.EPOCH_COMPLETED)
    # pylint: disable=unused-argument,unused-variable
    def validate(a_trainer):
        evaluators.train.run(data_loaders[0])
        evaluators.validation.run(data_loaders[1])

    @trainer.on(Events.COMPLETED)
    # pylint: disable=unused-argument,unused-variable
    def test(a_trainer):
        model.load_state_dict(torch.load(get_newest_file("checkpoints")))
        evaluators.test.run(data_loaders[2])

    add_early_stopping_and_checkpoint(
        evaluators.validation, trainer, "semigroup", model
    )
    with get_tensorboard_logger(trainer, evaluators, list(metrics.keys())):
        trainer.run(data_loaders[0], max_epochs=params["epochs"])
