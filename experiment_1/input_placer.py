"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Copyright (C) 2020 Lulof Pirée, Teun Schilperoort

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

from experiment_1.node import Marble

class InputPlacer(abc.ABC):
    """
    Abstract class providing the interface for classes
    that assign a set of input values a position in the network.
    """

    def __init__(self, input_pos: np.array, input_region_sizes: np.array):
        """
        Arguments:

        * input_pos: vertex of input region with lowest distance to the origin
            (lower left corner in 2D, generalizes to higher dimensions)
        * input_region_sizes: array of lengths of input region per dimension
            (in 2D, this are the width and the height respectively)
        """
        self.__input_pos = input_pos
        self.__input_region_sizes = input_region_sizes

    @property
    def input_pos(self):
        return self.__input_pos

    @property
    def input_region_sizes(self):
        return self.__input_region_sizes

    @abc.abstractmethod
    def marblize_data(self, input_data: Iterable[object]) -> Iterable[Marble]:
        """
        Given a set of data, create a Marble for each input datum,
        and assign them a position in the input region.
        """
        pass

class NewtonRaphsonInputPlacer(InputPlacer):
    #TODO: docstring

    def marblize_data(self, input_data: Iterable[object]) -> Iterable[Marble]:
        #TODO: docstring
        pass