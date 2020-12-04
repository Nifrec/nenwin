"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Unit-tests for Marble of node.py.
"""
import unittest
import numpy as np

from experiment_1.attraction_functions.attraction_functions import Gratan
from experiment_1.node import Marble
from experiment_1.particle import PhysicalParticle
from experiment_1.auxliary import generate_stiffness_dict
from test_aux import ATTRACT_FUNCT
from test_aux import NUMERICAL_ABS_ACCURACY_REQUIRED
from test_aux import check_close
from test_aux import ZERO


class MarbleTestCase(unittest.TestCase):

    def test_datum(self):
        datum = "Hello world"
        m = Marble(ZERO, ZERO, ZERO, 0, None, datum, 0, 0, 0, 0)
        self.assertEqual(m.datum, datum)

    def test_copy(self):
        pos = np.array([1])
        vel = np.array([2])
        acc = np.array([3])
        mass = 4
        datum = 5
        attraction_funct = ATTRACT_FUNCT
        stiffnesses = generate_stiffness_dict(0.6, 0.7, 0.8, 0.9)
        original = Marble(pos, vel, acc, mass, attraction_funct,
                          datum=datum, **stiffnesses)
        copy = original.copy()

        self.assertFalse(copy is original)

        self.assertTrue(check_close(acc, copy.acc))
        self.assertTrue(check_close(vel, copy.vel))
        self.assertTrue(check_close(pos, copy.pos))
        self.assertEqual(mass, copy.mass)
        self.assertTrue(attraction_funct is copy._attraction_function)
        self.assertEqual(copy.marble_stiffness,
                         stiffnesses["marble_stiffness"])
        self.assertEqual(copy.node_stiffness, stiffnesses["node_stiffness"])
        self.assertEqual(copy.marble_attraction,
                         stiffnesses["marble_attraction"])
        self.assertEqual(copy.node_attraction, stiffnesses["node_attraction"])
        self.assertEqual(copy.datum, datum)


if __name__ == '__main__':
    unittest.main()
