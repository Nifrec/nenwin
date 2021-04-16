"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
Februari 2021

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

InputPlacer variant that maps a 2D array (matrix) to a set of Marbles
positioned in a grid.
Also generalizes to higher dimensions.
The grid entried correspond to the entries 
in the matrix at the corresponding index.
"""
from typing import Iterable
import numpy as np
import torch

from nenwin.input_placer import InputPlacer
from nenwin.all_particles import Marble

class GridInputPlacer(InputPlacer):
    """
    InputPlacer variant that maps a 2D array (matrix) to a set of Marbles
    positioned in a regular grid.
    Also generalizes to higher dimensions.
    The grid entried correspond to the entries 
    in the matrix at the corresponding index.

    This most basic GridInputPlacer only looks at position,
    and ignores the values of the input-array entries.
    Subclasses can override hooks to make a more complex mapping.
    """

    def marblelize_data(self, input_data: Iterable[object]) -> Iterable[Marble]:
        return super().marblelize_data(input_data)

    def _map_value_to_mass(self, value: float) -> float:
        """
        Hook method used by marblelize_data() 
        to find the *mass* of the Marble
        that is to be associated with with a certain entry (value)
        in the given input_data.
        """
        return 1.0

    def _map_value_to_vel(self, value: float) -> torch.Tensor:
        """
        Hook method used by marblelize_data() 
        to find the *velocity vector* of the Marble
        that is to be associated with with a certain entry (value)
        in the given input_data.
        """
        return torch.zeros(self.num_dims, dtype=torch.float)

    def _map_value_to_acc(self, value: float) -> torch.Tensor:
        """
        Hook method used by marblelize_data() 
        to find the *acceleration vector* of the Marble
        that is to be associated with with a certain entry (value)
        in the given input_data.
        """
        return torch.zeros(self.num_dims, dtype=torch.float)

class MassGridInputPlacer(GridInputPlacer):
    """
    Variant of GridInputPlacer that uses the value of the
    corresponding input-data element to set the mass
    of a Marble.
    """

    def _map_value_to_mass(self, value: float) -> float:
        """
        Hook method used by marblelize_data() 
        to find the *mass* of the Marble
        that is to be associated with with a certain entry (value)
        in the given input_data.
        """
        return value