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
from __future__ import annotations
from typing import Iterable, Sequence, Set
import numpy as np
import torch
import itertools

from nenwin.input_placer import InputPlacer
from nenwin.all_particles import Marble
from nenwin.attraction_functions.attraction_functions import ThresholdGravity
from nenwin.auxliary import generate_stiffness_dict

GRID_MARBLE_STIFFNESSES = generate_stiffness_dict(marble_stiffness=1,
                                                  node_stiffness=0,
                                                  marble_attraction=0,
                                                  node_attraction=1)


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

    The Marbles have the following fixed properties:
        * ThresholdGravity attraction function.
        * marble_stiffness = 1
        * node_stiffness = 0
        * marble_attraction = 0
        * node_attraction = 1

    And the following properties are based on input values, 
    in a mapping defined by hook methods:
        * mass (default: 1)
        * vel (default: zero vector)
        * acc (default: zero vector)
        * threshold-gravity radius (default: 100)
    """

    def marblelize_data(self, input_data: np.ndarray | torch.Tensor
                        ) -> Iterable[Marble]:
        grid_cell_size = self.input_region_sizes / input_data.shape

        positions_per_dim = []

        for dim in range(self.num_dims):
            offset = self.input_pos[dim] + grid_cell_size[dim] / 2
            positions = [offset + grid_cell_size[dim]*cell
                         for cell in range(input_data.shape[dim])]
            positions_per_dim.append(positions)

        positions = itertools.product(*positions_per_dim)
        return self.__create_marbles(positions, input_data.reshape(-1))

    def __create_marbles(self, positions: Iterable[Sequence[float]],
                         values: Iterable[float]) -> Set[Marble]:
        marbles = set()
        for pos, value in zip(positions, values):
            pos = torch.tensor(pos, dtype=torch.float)
            vel = self._map_value_to_vel(value)
            acc = self._map_value_to_acc(value)
            mass = self._map_value_to_mass(value)
            attract_radius = self._map_value_to_gravity_radius(value)
            marble = Marble(pos, vel, acc, mass,
                            ThresholdGravity(attract_radius),
                            datum=value,
                            **GRID_MARBLE_STIFFNESSES
                            )
            marbles.add(marble)
        return marbles

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

    def _map_value_to_gravity_radius(self, value: float) -> float:
        """
        Hook method used by marblelize_data()
        to find the *gravity threshold radius* of the Marble's
        ThresholdGravityAttractionFunction.
        So that the radius corresponds correctly to a certain entry (value)
        in the given input_data.
        """
        return 100.0


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


class NoGradMassGridInputPlacer(MassGridInputPlacer):
    """
    Same as MassGridInputPlaces, but sets .requires_grad() to False
    for all generated Marbles.
    """

    def marblelize_data(self, input_data: np.ndarray | torch.Tensor
                        ) -> Iterable[Marble]:
        output = super().marblelize_data(input_data)
        for marble in output:
            marble.requires_grad_(False)
        return output


class VelInputPlacer(GridInputPlacer):
    """
    Variant of GridInputPlacer that uses the value of the
    corresponding input-data element to set the velocity of the Marble.
    Each element in the velocity tensor will have the same value.
    """

    def _map_value_to_vel(self, value: float) -> torch.Tensor:
        """
        Hook method used by marblelize_data()
        to find the *mass* of the Marble
        that is to be associated with with a certain entry (value)
        in the given input_data.
        """
        return value * torch.ones(self.num_dims, dtype=torch.float)