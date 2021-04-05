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


Class to record values obtained during training of a machine learning
algorithm.
"""

from typing import Tuple

class TrainingStats:
    """
    Class to record values obtained during training 
    of a machine learning algorithm.

    These values are:
        * training set loss
        * validation set loss
        * validation set accuracy
        * test set loss
        * test set accuracy
    """

    def __init__(self):
        self.__train_losses = []
        self.__vali_losses = []
        self.__vali_accuracies = []
        self.__test_losses = []
        self.__test_accuracies = []

    @property
    def train_losses(self) -> Tuple[float]:
        return tuple(self.__train_losses)

    @property
    def validation_losses(self) -> Tuple[float]:
        return tuple(self.__vali_losses)

    @property
    def validation_accuracies(self) -> Tuple[float]:
        return tuple(self.__vali_accuracies)

    @property
    def test_losses(self) -> Tuple[float]:
        return tuple(self.__test_losses)

    @property
    def test_accuracies(self) -> Tuple[float]:
        return tuple(self.__test_accuracies)

    def add_train_loss(self, loss: float):
        self.__train_losses.append(loss)

    def add_validation_loss(self, loss: float):
        self.__vali_losses.append(loss)

    def add_validation_accuracy(self, accuracy: float):
        self.__vali_accuracies.append(accuracy)

    def add_test_loss(self, loss: float):
        self.__test_losses.append(loss)

    def add_test_accuracy(self, accuracy: float):
        self.__test_accuracies.append(accuracy)