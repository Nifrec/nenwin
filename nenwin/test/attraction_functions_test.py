"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Copyright (C) 2020 Lulof Pirée, 

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

Unit-tests for nenwin/attraction_functions/attraction_functions.py
"""
import unittest
import numpy as np
import torch

from nenwin.particle import PhysicalParticle
from nenwin.attraction_functions.attraction_functions import Gratan
from nenwin.attraction_functions.attraction_functions \
    import NewtonianGravity
from nenwin.attraction_functions.attraction_functions \
    import ThresholdGravity
from nenwin.attraction_functions.attraction_functions \
    import TrainableThresholdGravity
from nenwin.test.test_aux import ZERO


class GratanTestCase(unittest.TestCase):
    def setUp(self):
        self.gratan = Gratan()

    def test_gratan_1(self):
        """
        Base case: both particles unit masses.
        """
        mass_1 = 1
        mass_2 = 1
        pos_1 = torch.tensor([1, 1], dtype=torch.float)
        pos_2 = torch.tensor([0, 0], dtype=torch.float)

        radius = torch.norm(pos_1 - pos_2)
        expected_attraction = 1 * (1 - abs(np.tanh(radius)))

        result = attraction_2_points(pos_1, mass_1, pos_2,
                                     mass_2, self.gratan)
        self.assertTrue(torch.allclose(expected_attraction, result))

    def test_gratan_2(self):
        """
        Corner case: zero radius.
        """
        mass_1 = 2
        mass_2 = 2

        pos_1 = torch.tensor([0, 0], dtype=torch.float)
        pos_2 = torch.tensor([0, 0], dtype=torch.float)

        expected_attraction = mass_1*mass_2

        result = attraction_2_points(pos_1, mass_1, pos_2,
                                     mass_2, self.gratan)
        self.assertAlmostEqual(expected_attraction, result)

    def test_gratan_3(self):
        """
        Base case: particles different masses.
        """
        mass_1 = 2
        pos_1 = torch.tensor([0, 0], dtype=torch.float)

        mass_2 = 32
        pos_2 = torch.tensor([0, 0], dtype=torch.float)

        radius = torch.norm(pos_1 - pos_2)
        expected_attraction = mass_1*mass_2 * (1 - abs(np.tanh(radius)))

        result = attraction_2_points(pos_1, mass_1, pos_2,
                                     mass_2, self.gratan)
        self.assertAlmostEqual(expected_attraction, result)

    def test_gratan_4(self):
        """
        Base case: Gratan with multiplier.
        """
        mass_1 = 2
        pos_1 = torch.tensor([0, 0], dtype=torch.float)
        mass_2 = 32
        pos_2 = torch.tensor([0, 0], dtype=torch.float)

        multiplier = 2
        attraction_function = Gratan(multiplier)

        radius = torch.norm(pos_1 - pos_2)
        expected_attraction = multiplier*mass_1*mass_2 \
            * (1 - abs(np.tanh(radius)))

        
        result = attraction_2_points(pos_1, mass_1, pos_2,
                                     mass_2, attraction_function)
        self.assertAlmostEqual(expected_attraction, result)

class NewtonainTestCase(unittest.TestCase):
    def setUp(self):
        self.newtonian = NewtonianGravity()

    def test_newtonian_1(self):
        """
        Base case: both particles unit masses.
        """
        pos_1 = torch.tensor([1, 1], dtype=torch.float)
        pos_2 = torch.tensor([0, 0], dtype=torch.float)
        mass_1 = 1
        mass_2 = 1

        radius = torch.norm(pos_1 - pos_2)
        expected_attraction = mass_1 * mass_2 / (radius**2)

        result = attraction_2_points(pos_1, mass_1, pos_2,
                                     mass_2, self.newtonian)
        self.assertAlmostEqual(expected_attraction, result)

    def test_newtonian_2(self):
        """
        Base case: one particle negative mass.
        """
        pos_1 = torch.tensor([1, 1], dtype=torch.float)
        pos_2 = torch.tensor([0, 0], dtype=torch.float)
        mass_1 = -1
        mass_2 = 1

        radius = torch.norm(pos_1 - pos_2)
        expected_attraction = mass_1 * mass_2 / (radius**2)

        result = attraction_2_points(pos_1, mass_1, pos_2,
                                     mass_2, self.newtonian)
        self.assertAlmostEqual(expected_attraction, result)

    def test_newtonian_3(self):
        """
        Corner case: zero radius.
        Infinite gravity force!
        """
        pos_1 = torch.tensor([1, 1], dtype=torch.float)
        pos_2 = torch.tensor([1, 1], dtype=torch.float)
        mass_1 = 1
        mass_2 = 1

        radius = torch.norm(pos_1 - pos_2)
        expected_attraction = float("inf")

        result = attraction_2_points(pos_1, mass_1, pos_2,
                                     mass_2, self.newtonian)
        self.assertAlmostEqual(expected_attraction, result)

    def test_newtonian_4(self):
        """
        Base case: large distance, large negative masses.
        """
        pos_1 = torch.tensor([1, 1], dtype=torch.float)
        pos_2 = torch.tensor([-100, -100], dtype=torch.float)
        mass_1 = -100
        mass_2 = -200

        radius = torch.norm(pos_1 - pos_2)
        expected_attraction = mass_1 * mass_2 / (radius**2)

        result = attraction_2_points(pos_1, mass_1, pos_2,
                                     mass_2, self.newtonian)
        self.assertTrue(torch.allclose(expected_attraction, result))

    def test_repr(self):
        self.assertEqual("NewtonianGravity()", repr(NewtonianGravity()))

class ThresholdGravityTestCase(unittest.TestCase):
    def setUp(self):
        self.threshold = 10
        self.threshold_grav = ThresholdGravity(10)

    def test_threshold_gravity_1(self):
        """
        Base case: below threshold.
        """
        pos_1 = torch.tensor([0, 0], dtype=torch.float)
        pos_2 = torch.tensor([1, 0], dtype=torch.float)
        mass = 1

        result = attraction_2_points(pos_1, mass, pos_2, mass, self.threshold_grav)
        expected = 1
        self.assertAlmostEqual(expected, result)

    def test_threshold_gravity_2(self):
        """
        Base case: above threshold.
        """
        pos_1 = torch.tensor([0, 0], dtype=torch.float)
        pos_2 = torch.tensor([11, 0], dtype=torch.float)
        mass = 1

        result = attraction_2_points(pos_1, mass, pos_2, mass, self.threshold_grav)
        expected = 0
        self.assertAlmostEqual(expected, result)

    def test_threshold_gravity_3(self):
        """
        Corner case: exactly at threshold.
        """
        pos_1 = torch.tensor([0, 0], dtype=torch.float)
        pos_2 = torch.tensor([10, 0], dtype=torch.float)
        mass = 1

        result = attraction_2_points(pos_1, mass, pos_2, mass, self.threshold_grav)
        expected = 1 / (10**2)
        self.assertAlmostEqual(expected, result)

    def test_repr(self):
        threshold = 4.0
        threshold_funct = ThresholdGravity(threshold)
        expected = f"ThresholdGravity(4.0)"
        self.assertEqual(expected, repr(threshold_funct))

class TrainableThresholdGravityTestCase(unittest.TestCase):

    def test_parameters(self):
        threshold = 13
        funct = TrainableThresholdGravity(threshold)
        params = tuple(funct.named_parameters())
        self.assertEqual(params[0][0], '_TrainableThresholdGravity__threshold')
        self.assertEqual(params[0][1], threshold)

    def test_repr(self):
        threshold = 4.0
        threshold_funct = TrainableThresholdGravity(threshold)
        expected = f"ThresholdGravity(4.0)"
        self.assertEqual(expected, repr(threshold_funct))

def attraction_2_points(pos_1: torch.Tensor,
                        mass_1: float,
                        pos_2: torch.Tensor,
                        mass_2: float,
                        attraction_funct: callable) -> float:
    zero = torch.tensor([0, 0], dtype=torch.float)
    p1 = PhysicalParticle(pos=pos_1,
                          vel=zero,
                          acc=zero,
                          mass=mass_1,
                          attraction_function=None)
    p2 = PhysicalParticle(pos=pos_2,
                          vel=zero,
                          acc=zero,
                          mass=mass_2,
                          attraction_function=None)

    return attraction_funct(p1, p2)

if __name__ == '__main__':
    unittest.main()
