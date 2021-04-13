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


Abstraction for an iterable dataset,
combining the train, test and validation set.
"""
from __future__ import annotations
import abc
from typing import Any, Iterable, Iterator, Tuple

from nenwin.all_particles import Marble

class Sample:
    """
    Record type for a input - expected label combination.
    For training or evaualuating a classification model.
    """

    def __init__(self, inputs: Iterable[float], label: int):
        self.inputs = inputs
        self.label = label

    def __getitem__(self, key: int) -> Iterable[float] | int:
        if key == 0:
            return self.inputs
        if key == 1:
            return self.label
        else:
            raise IndexError(f"Invalid index ´{key}´")

class Dataset(abc.ABC):
    """
    Abstraction for an iterable dataset for a classification task.
    Combines the train, test and validation set.
    """

    @abc.abstractmethod
    def iter_train(self) -> Iterator[Sample]:
        """
        Iterate one epoch over train set.
        Samples are returned in a randomized order.
        """
        ...

    @abc.abstractmethod
    def iter_validation(self) -> Iterator[Sample]:
        """
        Iterate one epoch over the validation set.
        """
        ...

    @abc.abstractmethod
    def iter_test(self) -> Iterator[Sample]:
        """
        Iterate one epoch over the test set.
        """
        ...

    @abc.abstractmethod
    def get_len_train(self) -> int: ...

    @abc.abstractmethod
    def get_len_validation(self) -> int: ...

    @abc.abstractmethod
    def get_len_test(self) -> int: ...

