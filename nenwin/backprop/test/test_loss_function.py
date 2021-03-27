"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
March 2021

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

Testcases for the source file loss_function.py
"""
from typing import List, Sequence
import torch
import unittest

from nenwin.marble_eater_node import MarbleEaterNode
from nenwin.node import Marble, Node
from nenwin.model import NenwinModel
from nenwin.auxliary import distance
from nenwin.backprop.loss_function import find_closest_marble_to

ZERO = torch.tensor([0.0, 0.0])


class FindClosestMarbleToTestCase(unittest.TestCase):

    def test_find_closest_marble_to_1(self):
        """
        Base case: target particle not in Model.
        """
        marble_positions = (
            (10, 10),
            (5, 10),
            (12.9, 3.2),
            (9.9, -0.7)
        )

        marbles = [Marble(torch.tensor(pos, dtype=torch.float),
                          ZERO, ZERO, 0, None, None)
                   for pos in marble_positions]
        model = NenwinModel([], marbles)

        target = Marble(torch.tensor([12.87, 2.9]), ZERO, ZERO, 0, None, None)

        expected = marbles[2]

        self.assertIs(find_closest_marble_to(target, model), expected)

    def test_find_closest_marble_to_2(self):
        """
        Base case: target particle in Model.
        """
        marble_positions = (
            (10, 10),
            (5, 10),
            (12.9, 3.2),
            (9.9, -0.7),
            (11.1, 3.5)
        )

        marbles = [Marble(torch.tensor(pos, dtype=torch.float),
                          ZERO, ZERO, 0, None, None)
                   for pos in marble_positions]
        model = NenwinModel([], marbles)

        target = marbles[2]

        expected = marbles[4]

        self.assertIs(find_closest_marble_to(target, model), expected)

    def test_error_if_no_other_marbles(self):
        """
        Cannot find closest Marble if no other Marbles exist.
        """
        target = Marble(torch.tensor([12.87, 2.9]), ZERO, ZERO, 0, None, None)
        model = NenwinModel([], [target])

        with self.assertRaises(RuntimeError):
            find_closest_marble_to(target, model)


if __name__ == '__main__':
    unittest.main()
