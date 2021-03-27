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

Testcases for the source file loss_function.py
"""
from typing import List, Sequence
import torch
import unittest

from nenwin.marble_eater_node import MarbleEaterNode
from nenwin.node import Marble, Node
from nenwin.model import NenwinModel
from nenwin.auxliary import distance
from nenwin.backprop.loss_function import find_closest_marble_to, find_most_promising_marble_to, \
    velocity_weighted_distance

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


class FindMostPromisingMarbleTo(unittest.TestCase):

    def test_find_most_promising_marble_to_1(self):
        """
        Base case: target particle not in Model. Both weights are 1.
        """
        marble_positions = (
            (0, 0),
            (10.2, 10.1),
            (5, 10),
            (12.9, 3.2),
            (9.9, -0.7)
        )

        marble_velocities = (
            (0, 0),
            (5, 5.1),
            (10, -100),
            (1.2, 1),
            (0, 0)
        )

        marbles = [Marble(torch.tensor(pos, dtype=torch.float),
                          torch.tensor(vel, dtype=torch.float),
                          ZERO, 0, None, None)
                   for pos, vel in zip(marble_positions, marble_velocities)]
        model = NenwinModel([], marbles)

        target = Marble(torch.tensor([15.0, 15.0]), ZERO, ZERO, 0, None, None)

        expected = marbles[1]

        self.assertIs(find_most_promising_marble_to(
            target, model, 1, 1), expected)

    def test_find_most_promising_marble_to_2(self):
        """
        Base case: target particle not in Model. Weights differ.
        """
        marble_positions = (
            (0, 0),
            (10.0, 10.0)
        )

        marble_velocities = (
            (7.5, 7.5),
            (5.0, 5.0)
        )

        marbles = [Marble(torch.tensor(pos, dtype=torch.float),
                          torch.tensor(vel, dtype=torch.float),
                          ZERO, 0, None, None)
                   for pos, vel in zip(marble_positions, marble_velocities)]
        model = NenwinModel([], marbles)

        target = Marble(torch.tensor([15.0, 15.0]), ZERO, ZERO, 0, None, None)

        expected = marbles[0]

        pos_weight = 1
        vel_weight = 2
        result = find_most_promising_marble_to(target, model,
                                               pos_weight, vel_weight)
        self.assertIs(result, expected)


class VelocityWeightedDistanceTestCase(unittest.TestCase):

    def test_velocity_weighted_distance_1(self):
        """
        Base case: 0 distance, both weights 1.
        """
        pos_1 = torch.tensor([15.0, 15.0])
        m1 = Marble(pos_1, ZERO, ZERO, 0, None, None)

        pos_2 = torch.tensor([0.0, 0.0])
        vel_2 = torch.tensor([15.0, 15.0])
        m2 = Marble(pos_2, vel_2, ZERO, 0, None, None)

        expected = torch.tensor(0.0)
        result = velocity_weighted_distance(m1, m2, 1, 1)

        torch.testing.assert_allclose(result, expected)

    def test_velocity_weighted_distance_2(self):
        """
        Base case: vel weight not zero.
        """
        pos_1 = torch.tensor([0.0, 15.0])
        m1 = Marble(pos_1, ZERO, ZERO, 0, None, None)

        pos_2 = torch.tensor([0.0, 0.0])
        vel_2 = torch.tensor([0.0, 30.0])
        m2 = Marble(pos_2, vel_2, ZERO, 0, None, None)

        expected = torch.tensor(25.0)
        pos_weight = 1
        vel_weight = 1/3
        result = velocity_weighted_distance(m1, m2, pos_weight, vel_weight)

        torch.testing.assert_allclose(result, expected)

    def test_velocity_weighted_distance_3(self):
        """
        Corner case: pos weight zero.
        """
        pos_1 = torch.tensor([0.0, 15.0])
        m1 = Marble(pos_1, ZERO, ZERO, 0, None, None)

        pos_2 = torch.tensor([0.0, 0.0])
        vel_2 = torch.tensor([0.0, 30.0])
        m2 = Marble(pos_2, vel_2, ZERO, 0, None, None)

        expected = torch.tensor(100.0)
        pos_weight = 0
        vel_weight = 1/3
        result = velocity_weighted_distance(m1, m2, pos_weight, vel_weight)

        torch.testing.assert_allclose(result, expected)

    def test_velocity_weighted_distance_3(self):
        """
        Corner case: vel weight zero.
        """
        pos_1 = torch.tensor([123, 15.0])
        m1 = Marble(pos_1, ZERO, ZERO, 0, None, None)

        pos_2 = torch.tensor([4.3, 2.4])
        vel_2 = torch.tensor([89.0, 30.0])
        m2 = Marble(pos_2, vel_2, ZERO, 0, None, None)

        expected = distance(m1, m2)**2
        pos_weight = 1
        vel_weight = 0
        result = velocity_weighted_distance(m1, m2, pos_weight, vel_weight)

        torch.testing.assert_allclose(result, expected)


if __name__ == '__main__':
    unittest.main()
