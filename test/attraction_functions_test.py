"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Unit-tests for experiment_1/attraction_functions/attraction_functions.py
"""
import unittest
import numpy as np
from experiment_1.particle import PhysicalParticle
from experiment_1.attraction_functions.attraction_functions import Gratan
from experiment_1.attraction_functions.attraction_functions \
    import NewtonianGravity
from test_aux import check_close


class GratanTestCase(unittest.TestCase):
    def setUp(self):
        self.gratan = Gratan()

    def test_gratan_1(self):
        """
        Base case: both particles unit masses.
        """
        zero = np.array([0, 0])
        mass = 1

        pos_1 = np.array([1, 1])
        p1 = PhysicalParticle(pos=pos_1, vel=zero, acc=zero,
                              mass=mass, attraction_function=self.gratan)

        pos_2 = np.array([0, 0])
        p2 = PhysicalParticle(pos=pos_2, vel=zero, acc=zero,
                              mass=mass, attraction_function=self.gratan)

        radius = np.linalg.norm(pos_1 - pos_2)
        expected_attraction = 1 * (1 - abs(np.tanh(radius)))

        result = self.gratan(p1, p2)
        self.assertTrue(check_close(expected_attraction, result))

    def test_gratan_2(self):
        """
        Corner case: zero radius.
        """
        zero = np.array([0, 0])
        mass = 2

        pos_1 = np.array([0, 0])
        p1 = PhysicalParticle(pos=pos_1, vel=zero, acc=zero,
                              mass=mass, attraction_function=self.gratan)

        pos_2 = np.array([0, 0])
        p2 = PhysicalParticle(pos=pos_2, vel=zero, acc=zero,
                              mass=mass, attraction_function=self.gratan)

        expected_attraction = mass*mass

        result = self.gratan(p1, p2)
        self.assertTrue(check_close(expected_attraction, result))

    def test_gratan_3(self):
        """
        Base case: particles different masses.
        """
        zero = np.array([0, 0])

        mass_1 = 2
        pos_1 = np.array([0, 0])
        p1 = PhysicalParticle(pos=pos_1, vel=zero, acc=zero,
                              mass=mass_1, attraction_function=self.gratan)

        mass_2 = 32
        pos_2 = np.array([0, 0])
        p2 = PhysicalParticle(pos=pos_2, vel=zero, acc=zero,
                              mass=mass_2, attraction_function=self.gratan)

        radius = np.linalg.norm(pos_1 - pos_2)
        expected_attraction = mass_1*mass_2 * (1 - abs(np.tanh(radius)))

        result = self.gratan(p1, p2)
        self.assertTrue(check_close(expected_attraction, result))

    def test_gratan_4(self):
        """
        Base case: with multiplier.
        """
        multiplier = 2
        attraction_function = Gratan(multiplier)

        zero = np.array([0, 0])

        mass_1 = 2
        pos_1 = np.array([0, 0])
        p1 = PhysicalParticle(pos=pos_1,
                              vel=zero,
                              acc=zero,
                              mass=mass_1,
                              attraction_function=attraction_function)

        mass_2 = 32
        pos_2 = np.array([0, 0])
        p2 = PhysicalParticle(pos=pos_2,
                              vel=zero,
                              acc=zero,
                              mass=mass_2,
                              attraction_function=attraction_function)

        radius = np.linalg.norm(pos_1 - pos_2)
        expected_attraction = multiplier*mass_1*mass_2 \
            * (1 - abs(np.tanh(radius)))

        result = attraction_function(p1, p2)
        self.assertTrue(check_close(expected_attraction, result))


if __name__ == '__main__':
    unittest.main()
