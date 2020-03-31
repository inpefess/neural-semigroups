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
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy
from ignite.metrics.loss import Loss
from torch.nn import CrossEntropyLoss

from neural_semigroups import MagmaClassifier
from neural_semigroups.training_helpers import (get_arguments, get_loaders,
                                                learning_pipeline)


def train_classifier():
    """ train and save a classifier """
    args = get_arguments()
    magma_cardinality = args.cardinality
    classifier = MagmaClassifier(
        cardinality=magma_cardinality,
        hidden_dims=2 * [magma_cardinality ** 3],
    )
    loss = CrossEntropyLoss()
    evaluator = create_supervised_evaluator(
        classifier,
        metrics={"loss": Loss(loss), "accuracy": Accuracy()}
    )
    data_loaders = get_loaders(
        cardinality=magma_cardinality,
        batch_size=args.batch_size,
        train_size=args.train_size,
        validation_size=args.validation_size,
        use_labels=True
    )
    learning_pipeline(args, classifier, evaluator, loss, data_loaders)


if __name__ == "__main__":
    train_classifier()
