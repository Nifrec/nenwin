"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
April 2021

Copyright (C) 2021 Lulof Pirée, Teun Schilperoort

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


Class for managing the training of a NenwinModel using backpropagation.
"""

import torch
import matplotlib.axes as axes
import matplotlib.pyplot as plt
from typing import Callable, Iterable, Optional, Tuple
import time
import os

from nenwin.backprop.loss_function import NenwinLossFunction
from nenwin.model import NenwinModel
from nenwin.backprop.training_stats import TrainingStats
from nenwin.input_placer import InputPlacer
from nenwin.backprop.filename_gen import FilenameGenerator
class NenwinTrainer:
    """
    Class for managing the training of a NenwinModel using backpropagation.
    """

    def __init__(self,
                 model: NenwinModel,
                 loss_funct: NenwinLossFunction,
                 optimizer: torch.optim.Optimizer,
                 name_gen: FilenameGenerator,
                 input_places: InputPlacer):
        ...

    def run_training(self,
                     num_iters: int,
                     trainset_iter_funct: Callable,
                     validationset_iter_funct: Optional[Callable] = None,
                     checkpoint_interval: int = 1,
                     ):
        ...

    def evaluate_on_datasetset(self, dataset_iter: Iterable
                            ) -> Tuple[float, float]:
        """
        Evaluate the current performance of the model
        on one epoch of the given dataset.
        This dataset can for example the testset or the validation set.

        Arguments:
            * dataset_iter: Iterable that iterates over all samples
                of the dataset once.

        Returns:
            * accuracy: fraction of correct predictions
            * loss: sum of losses of each prediction.
        """
        ...

    def visualize_model(self) -> axes.Axes:
        ...

    def reset_stats(self):
        ...

    @property
    def training_stats(self) -> TrainingStats:
        ...

    @property
    def model(self) -> NenwinModel:
        ...

    def get_current_model_output(self) -> int:
        ...
