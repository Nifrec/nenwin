"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
May 2021

Copyright (C) 2021 Lulof Pirée

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

Class to place a set of input values into a given space.
"""
import abc
import numpy as np
from typing import Iterable
import torch

from nenwin.node import Marble

ZERO = torch.tensor([0, 0], dtype=torch.float)


class InputPlacer(abc.ABC):
    """
    Abstract class providing the interface for classes
    that assign a set of input values a position in the network.
    """

    def __init__(self, input_pos: np.ndarray, input_region_sizes: np.ndarray):
        """
        Arguments:

        * input_pos: vertex of input region with lowest distance to the origin
            (lower left corner in 2D, generalizes to higher dimensions,
            assuming the whole input region is in the positive subspace).
        * input_region_sizes: array of lengths of input region per dimension
            (in 2D, this are the width and the height respectively).
        """
        self.__input_pos = np.array(input_pos)
        self.__input_region_sizes = np.array(input_region_sizes)
        self.__num_dims = len(input_region_sizes)

    @property
    def input_pos(self) -> np.ndarray:
        return self.__input_pos.copy()

    @property
    def input_region_sizes(self) -> np.ndarray:
        return self.__input_region_sizes.copy()

    @property
    def num_dims(self) -> int:
        return self.__num_dims

    @abc.abstractmethod
    def marblelize_data(self, input_data: Iterable[object]) -> Iterable[Marble]:
        """
        Given a set of data, create a Marble for each input datum,
        and assign them a position in the input region.
        """
        pass
