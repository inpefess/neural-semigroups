""""
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
# pylint: disable-all
import sys
from unittest import TestCase
from unittest.mock import patch

import torch
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import RunningAverage, Loss
from torch import Tensor
from torch.nn import Linear, Module, Sequential
from torch.nn.functional import kl_div
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from neural_semigroups.constant_baseline import CURRENT_DEVICE
from neural_semigroups.magma import Magma
from neural_semigroups.training_helpers import (
    ThreeEvaluators,
    add_early_stopping_and_checkpoint,
    associative_ratio,
    get_arguments,
    get_loaders,
    get_tensorboard_logger,
    get_trainer,
    guessed_ratio,
    learning_pipeline,
    load_database_as_cubes,
)
from neural_semigroups.utils import FOUR_GROUP


class TestTrainingHelpers(TestCase):
    def setUp(self):
        torch.manual_seed(47)

    def test_get_arguments(self):
        args = "program --cardinality 2 --train_size 1 --validation_size 1"
        with patch.object(sys, "argv", args.split(" ")):
            get_arguments()

    @patch("neural_semigroups.training_helpers.load_database_as_cubes")
    def test_get_loaders(self, load_database_as_cubes_mock):
        train = torch.tensor([1])
        train_labels = torch.tensor([2])
        validation = torch.tensor([3])
        validation_labels = torch.tensor([4])
        test = torch.tensor([5])
        test_labels = torch.tensor([6])
        load_database_as_cubes_mock.return_value = (
            train,
            validation,
            test,
            train_labels,
            validation_labels,
            test_labels,
        )
        train_loader, validation_loader, test_loader = get_loaders(
            "filename", 1, 1, 1, 0.0
        )
        batch = [i for i in train_loader][0]
        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), 2)
        self.assertTrue(torch.allclose(batch[0], train))
        self.assertTrue(torch.allclose(batch[1], train_labels))
        batch = [i for i in validation_loader][0]
        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), 2)
        self.assertTrue(torch.allclose(batch[0], validation))
        self.assertTrue(torch.allclose(batch[1], validation_labels))
        batch = [i for i in test_loader][0]
        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), 2)
        self.assertTrue(torch.allclose(batch[0], test))
        self.assertTrue(torch.allclose(batch[1], test_labels))

    def test_load_database_as_cubes(self):
        half_result = [
            torch.tensor(
                [
                    [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]],
                    [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                ]
            ),
            torch.tensor(
                [
                    [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]],
                    [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                ]
            ),
            torch.tensor(
                [
                    [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]],
                    [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
                    [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                    [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]],
                ]
            ),
        ]
        true_result = half_result + half_result
        result = load_database_as_cubes(2, 1, 1, 0.0)
        for i, tensor in enumerate(result):
            self.assertTrue(tensor.allclose(true_result[i]))

    def test_associate_ratio(self):
        ratio = associative_ratio(
            Magma(FOUR_GROUP).probabilistic_cube.view(-1, 4, 4, 4),
            torch.tensor(0.0),
        )
        self.assertTrue(torch.tensor(1.0).allclose(ratio))

    def test_guessed_ratio(self):
        ratio = guessed_ratio(
            Magma(FOUR_GROUP).probabilistic_cube.view(-1, 4, 4, 4),
            Magma(FOUR_GROUP).probabilistic_cube.view(-1, 4, 4, 4),
        )
        self.assertTrue(torch.tensor(1.0).allclose(ratio))

    def test_add_early_stopping_and_checkpoint(self):
        module = Module()
        evaluator = Engine(lambda x, y: 0.0)
        trainer = Engine(lambda x, y: 0.0)
        add_early_stopping_and_checkpoint(evaluator, trainer, "test", module)
        completed_handlers = evaluator._event_handlers[Events.COMPLETED]
        self.assertEqual(len(completed_handlers), 2)
        self.assertIsInstance(completed_handlers[0][0], EarlyStopping)
        self.assertIsInstance(completed_handlers[1][0], ModelCheckpoint)

    def test_get_trainer(self):
        trainer = get_trainer(Sequential(Linear(1, 1)), 1.0, Loss(lambda x: x))
        print(trainer._event_handlers[Events.ITERATION_COMPLETED])
        self.assertIsInstance(
            trainer._event_handlers[Events.ITERATION_COMPLETED][1][0].__self__,
            RunningAverage,
        )
        self.assertEqual(
            trainer._event_handlers[Events.ITERATION_COMPLETED][1][1][1],
            "running_loss",
        )
        self.assertIsInstance(
            trainer._event_handlers[Events.EPOCH_COMPLETED][0][1][1],
            ProgressBar,
        )

    def test_get_tensorboard_logger(self):
        trainer = Engine(lambda x, y: 0.0)
        three_evaluators = ThreeEvaluators(Engine(lambda x, y: 0.0), dict())
        tensorboard_logger = get_tensorboard_logger(
            trainer, three_evaluators, []
        )
        self.assertEqual(
            trainer._event_handlers[Events.EPOCH_COMPLETED][0][1][1],
            tensorboard_logger,
        )
        for engine in [
            three_evaluators.validation,
            three_evaluators.test,
        ]:
            self.assertEqual(
                engine._event_handlers[Events.COMPLETED][0][1][1],
                tensorboard_logger,
            )

    def test_learning_pipeline(self):
        data_loaders = 3 * [
            DataLoader(
                TensorDataset(
                    torch.ones([1, 2, 2, 2]).to(CURRENT_DEVICE),
                    torch.ones([1, 2, 2, 2]).to(CURRENT_DEVICE),
                )
            )
        ]

        class NewModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(torch.nn.Linear(2, 2)).to(
                    CURRENT_DEVICE
                )

            def forward(self, x):
                return self.layers(x)

        def loss(prediction: Tensor, target: Tensor) -> Tensor:
            return kl_div(torch.log(prediction), target, reduction="batchmean")

        model = NewModel()
        before = next(model.parameters()).detach()
        learning_pipeline(
            params={"learning_rate": 1.0, "epochs": 1},
            model=model,
            loss=loss,
            metrics={"loss": Loss(loss)},
            data_loaders=data_loaders,
        )
        after = next(model.parameters()).detach()
        self.assertTrue(torch.allclose(before, after))
