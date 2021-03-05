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

Unit-tests for aux.py.
"""
import unittest
import numpy as np

from nenwin.particle import Particle
from nenwin.test.test_aux import check_close
from nenwin.auxliary import distance
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