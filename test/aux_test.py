"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Unit-tests for aux.py.
"""
import unittest
import numpy as np

from experiment_1.particle import Particle
from test_aux import check_close
from experiment_1.aux import distance
class DistanceCase(unittest.TestCase):

    def test_distance_1(self):
        """
        Base case: 2 nodes at different positions.
        """
        pos1 = np.array([1, 0, 0])
        pos2 = np.array([2, 0, 0])
        zero_vect = np.array([0, 0, 0])
        p1 = Particle(pos1, zero_vect, zero_vect)
        p2 = Particle(pos2, zero_vect, zero_vect)

        expected = 1

        self.assertTrue(check_close(distance(p1, p2), expected))

    def test_distance_2(self):
        """
        Corner case: 2 nodes at same positions.
        """
        pos1 = np.array([9, 8, 70.5])
        pos2 = np.array([9, 8, 70.5])
        zero_vect = np.array([0, 0, 0])
        p1 = Particle(pos1, zero_vect, zero_vect)
        p2 = Particle(pos2, zero_vect, zero_vect)

        expected = 0

        self.assertTrue(check_close(distance(p1, p2), expected))

    def test_distance_3(self):
        """
        Corner case: 2 nodes at different position that differ 
        in multiple dimensions.
        """
        pos1 = np.array([1, 1, 1])
        pos2 = np.array([-1, -1, -1])
        zero_vect = np.array([0, 0, 0])
        p1 = Particle(pos1, zero_vect, zero_vect)
        p2 = Particle(pos2, zero_vect, zero_vect)

        expected = np.sqrt(3*2**2)

        self.assertTrue(check_close(distance(p1, p2), expected))

if __name__ == '__main__':
    unittest.main()