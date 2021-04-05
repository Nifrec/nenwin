"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
March 2021

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


class FilenameGenerator:
    ...


class NenwinTrainer:

    def __init__(self,
                 model: NenwinModel,
                 loss_funct: NenwinLossFunction,
                 optimizer: torch.optim.Optimizer,
                 name_gen: FilenameGenerator):
        ...

    def run_training(self,
                     num_iters: int,
                     trainset_iter_funct: Callable,
                     validationset_iter_funct: Optional[Callable] = None,
                     checkpoint_interval: int = 1,
                     ):
        ...

    def evaluate_on_testset(self, testset_iter: Iterable
                            ) -> Tuple[float, float]:
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


class FilenameGenerator:
    """
    Class to generate filenames with a timestamp.
    """

    def __init__(self, directory: str, base: str, extension: str):
        """
        Arguments:
            * directory: path to directory containing the target file.
            * base: prefix of the name of the file itself.
            * extension: suffix of the file, 
                usually an extension such as ".pt".
        """
        self.__directory = directory
        self.__base = base
        self.__extension = extension

    def gen_filename(self, is_checkpoint: bool = False) -> str:
        if is_checkpoint:
            checkpoint = "_checkpoint"
        else:
            checkpoint = ""

        name = self.__base + time.asctime() + checkpoint + self.__extension
        output = os.path.joint(self.__directory, name)
        return output
