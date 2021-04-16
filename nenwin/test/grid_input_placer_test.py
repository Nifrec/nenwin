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

Unit-tests for GridInputPlacer from grid_input_placer.py.
"""
from typing import Iterable, Sequence, Set
import numpy as np
import unittest
import itertools
import torch

from nenwin.grid_input_placer import GridInputPlacer
from nenwin.all_particles import Marble


class GridInputPlacerTestCase2D(unittest.TestCase):

    def setUp(self):
        self.grid_pos = np.array([0, 0])
        self.grid_size = np.array([100, 50])
        self.input_placer = GridInputPlacer(self.grid_pos, self.grid_size)

    def test_num_dims(self):
        expected = 2
        result = self.input_placer.num_dims
        self.assertEqual(expected, result)

    def test_marbilize_data_4x4(self):
        """
        4x4 input matrix should produce 16 Marbles
        at deterministic positions in the grid.
        """
        input_size = (4, 4)
        input_data = np.random.uniform(low=-10, high=10, size=input_size)

        expected_y_positions = [6.25 + 12.5 * row
                                for row in range(input_size[0])]
        expected_x_positions = [12.5 + 25 * col
                                for col in range(input_size[1])]

        expected_positions = tuple(itertools.product(expected_x_positions,
                                                     expected_y_positions))

        result = self.input_placer.marblelize_data(input_data)
        check_marbles_at_positions(expected_positions, result)

        self.assertTrue(all(isinstance(m, Marble) for m in result))

    def test_marbilize_data_1x1(self):
        """
        Corner case: smallest allowed input.
        """
        input_size = (1, 1)
        input_data = np.random.uniform(low=-10, high=10, size=input_size)
        expected_pos = (50, 25)
        result = self.input_placer.marblelize_data(input_data)
        check_marbles_at_positions([expected_pos], result)


class GridInputPlacerTestCase3D(unittest.TestCase):

    def setUp(self):
        self.grid_pos = np.array([10, 10, 20])
        self.grid_size = np.array([10, 10, 20])
        self.input_placer = GridInputPlacer(self.grid_pos, self.grid_size)

    def test_num_dims(self):
        expected = 3
        result = self.input_placer.num_dims
        self.assertEqual(expected, result)

    def test_marbilize_data_2x2x4(self):
        """
        4x4 input matrix should produce 16 Marbles
        at deterministic positions in the grid.
        """
        input_size = (2, 2, 4)
        input_data = np.random.uniform(low=-10, high=10, size=input_size)

        
        expected_x_positions = [10 + 2.5 + 5 * col
                                for col in range(input_size[1])]
        expected_y_positions = [10 + 2.5 + 5 * row
                                for row in range(input_size[0])]
        expected_z_positions = [20 + 2.5 + 5 * bar
                                for bar in range(input_size[2])]

        expected_positions = tuple(itertools.product(expected_x_positions,
                                                     expected_y_positions,
                                                     expected_z_positions))

        result = self.input_placer.marblelize_data(input_data)
        check_marbles_at_positions(expected_positions, result)

def check_marbles_at_positions(positions: Iterable[Sequence[float]],
                               marbles: Set[Marble]):
    """
    Assert that:
        (1) For each pos in positions,
            there is one Marble m in [marbles] with m.pos = pos
        (2) And vice versa
        (3) No two Marbles are mapped to the same position or vice versa.

    In other words:
        The mapping of Marble positions to [positions] is a bijection,
        with a relation only defined for equally valued entries.
        So if the same position occurs twice or more in one of the
        two multisets, they should also have the same number of occurrences
        in the other multisetset.
    """
    marbles = set(marbles)
    for pos in positions:
        has_match = False
        for marble in marbles:
            if torch.allclose(marble.pos, torch.tensor(pos, dtype=torch.float)):
                has_match = True
                marbles.remove(marble)
                break
        assert has_match, f"{pos} not in {list(map(lambda m:m.pos, marbles))}"

    assert len(marbles) == 0


if __name__ == "__main__":
    unittest.main()
