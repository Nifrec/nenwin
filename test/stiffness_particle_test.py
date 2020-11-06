"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Unit-tests for StiffnessParticle of particle.py.
"""
import unittest
import numpy as np

from experiment_1.stiffness_particle import StiffnessParticle
from experiment_1.attraction_functions.attraction_functions import Gratan
from experiment_1.node import Node
from experiment_1.marble import Marble
from experiment_1.particle import PhysicalParticle
from test_aux import TestAttractionFunction
from test_aux import NUMERICAL_ABS_ACCURACY_REQUIRED
from test_aux import check_close
from test_aux import runge_kutta_4_step

ZERO = np.array([0])
ATTRACT_FUNCT = TestAttractionFunction()


class StiffnessParticleTestCase(unittest.TestCase):

    def test_stiffness_getters(self):
        marble_stiffness = 0.42
        node_stiffness = 0.13
        particle = create_particle(
            marble_stiffness, node_stiffness, ZERO, ZERO)
        self.assertEqual(particle.marble_stiffness, marble_stiffness)
        self.assertEqual(particle.node_stiffness, node_stiffness)

    def test_attraction_getters(self):
        marble_attraction = 0.42
        node_attraction = 0.13
        particle = create_particle(ZERO, ZERO,
                                   marble_attraction, node_attraction)
        self.assertEqual(particle.marble_attraction, marble_attraction)
        self.assertEqual(particle.node_attraction, node_attraction)

    def test_stiffness_errors_1(self):
        self.assertRaises(ValueError, create_particle, 1.1, ZERO, ZERO, ZERO)

    def test_stiffness_errors_2(self):
        self.assertRaises(ValueError, create_particle, ZERO, 1.1, ZERO, ZERO)

    def test_stiffness_errors_3(self):
        self.assertRaises(ValueError, create_particle, ZERO, -0.1, ZERO, ZERO)

    def test_stiffness_errors_4(self):
        self.assertRaises(ValueError, create_particle, -1, ZERO, ZERO, ZERO)

    def test_attraction_errors_1(self):
        self.assertRaises(ValueError, create_particle, ZERO, ZERO, 1.1, ZERO)

    def test_attraction_errors_2(self):
        self.assertRaises(ValueError, create_particle, ZERO, ZERO, ZERO, 1.1)

    def test_attraction_errors_3(self):
        self.assertRaises(ValueError, create_particle, ZERO, ZERO, -1, ZERO)

    def test_attraction_errors_4(self):
        self.assertRaises(ValueError, create_particle, ZERO, ZERO, ZERO, -0.1)

    def test_attraction_node_1(self):
        """
        Check if the node_attraction multiplier is used to compute force to a
        node.
        """
        node_attraction = 0.5
        # Position -1 ensures it will be pulled towards '+' x-direction
        node = Node(np.array([-1]), ZERO, ZERO, 0, None, 0)
        particle = create_particle(0, 0, 0, node_attraction)

        result = particle.compute_attraction_force_to(node)
        self.assertEqual(result, ATTRACT_FUNCT.value * node_attraction)

    def test_attraction_node_2(self):
        """
        Corner case: should not attract any node if the multiplier is 0.
        """
        node_attraction = 0
        node = Node(np.array([-1]), ZERO, ZERO, 0, None, 0)
        particle = create_particle(0, 0, 0, node_attraction)
        result = particle.compute_attraction_force_to(node)
        self.assertEqual(result, 0)

    def test_attraction_marble_1(self):
        """
        Check if the marble_attraction multiplier is used to compute force to a
        node.
        """
        marble_attraction = 0.5
        # Position -1 ensures it will be pulled towards '+' x-direction
        marble = Marble(np.array([-1]), ZERO, ZERO, 0, None, None)
        particle = create_particle(0, 0, marble_attraction, 0)

        result = particle.compute_attraction_force_to(marble)
        self.assertEqual(result, ATTRACT_FUNCT.value * marble_attraction)

    def test_attraction_marble_2(self):
        """
        Corner case: should not attract any node if the multiplier is 0.
        """
        marble_attraction = 0
        marble = Marble(np.array([-1]), ZERO, ZERO, 0, None, None)
        particle = create_particle(0, 0, marble_attraction, 0)
        result = particle.compute_attraction_force_to(marble)
        self.assertEqual(result, 0)

    def test_attraction_wrong_type(self):
        """
        Corner case: can only compute attraction to Marbles and Nodes.
        """
        other = PhysicalParticle(np.array([-1]), ZERO, ZERO, 0, None)
        particle = create_particle(0, 0, 0, 0)
        self.assertRaises(
            ValueError, particle.compute_attraction_force_to, other)


def create_particle(marble_stiffness,
                    node_stiffness,
                    marble_attraction,
                    node_attraction) -> StiffnessParticle:
    """
    Simply attempt to create a StiffnessParticle with given parameters,
    and 0 or None for all other parameter values.
    """
    return StiffnessParticle(ZERO, ZERO, ZERO, ZERO, ATTRACT_FUNCT,
                             marble_stiffness=marble_stiffness,
                             node_stiffness=node_stiffness,
                             marble_attraction=marble_attraction,
                             node_attraction=node_attraction)


if __name__ == '__main__':
    unittest.main()
