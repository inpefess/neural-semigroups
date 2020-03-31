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
from ignite.engine import create_supervised_evaluator
from ignite.metrics.loss import Loss
from torch import Tensor
from torch.nn.functional import kl_div

from neural_semigroups.denoising_autoencoder import MagmaDAE
from neural_semigroups.training_helpers import (get_arguments, get_loaders,
                                                learning_pipeline)


def train_denoising_autoencoder():
    """ train and save a denoising autoencoder """
    args = get_arguments()
    cardinality = args.cardinality
    dae = MagmaDAE(
        cardinality=cardinality,
        hidden_dims=[
            cardinality ** 2,
            cardinality
        ],
        corruption_rate=0.5
    )

    def loss(prediction: Tensor, target: Tensor) -> Tensor:
        return kl_div(torch.log(prediction), target, reduction="batchmean")
    evaluator = create_supervised_evaluator(
        dae,
        metrics={"loss": Loss(loss)}
    )
    data_loaders = get_loaders(
        cardinality=cardinality,
        batch_size=args.batch_size,
        train_size=args.train_size,
        validation_size=args.validation_size,
        use_labels=False
    )
    learning_pipeline(args, dae, evaluator, loss, data_loaders)


if __name__ == "__main__":
    train_denoising_autoencoder()
