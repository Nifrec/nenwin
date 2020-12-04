"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Unit-tests for NenwinModel class of model.py.
"""
import unittest
import numpy as np
from typing import Tuple

from experiment_1.node import Node, Marble
from experiment_1.model import NenwinModel
from experiment_1.attraction_functions.attraction_functions \
    import AttractionFunction
from experiment_1.particle import PhysicalParticle
from experiment_1.marble_eater_node import MarbleEaterNode
from test_aux import check_close
from test_aux import TEST_SIMULATION_STEP_SIZE
from test_aux import runge_kutta_4_step
from test_aux import TestAttractionFunction
from test_aux import ZERO

ATTRACT_FUNCT = TestAttractionFunction()


class ModelTestCase(unittest.TestCase):

    def test_marbles_getter(self):
        """
        marbles() getter should return only and all marbles.
        """
        marble = generate_dummy_marble()
        node = generate_dummy_node()
        model = NenwinModel(set([node]), set([marble]))
        self.assertSetEqual(set([marble]), model.marbles)

    def test_nodes_getter(self):
        """
        nodes() getter should return only and all non-Marble Nodes.
        """
        marble = generate_dummy_marble()
        node = generate_dummy_node()
        model = NenwinModel(set([node]), set([marble]))
        self.assertSetEqual(set([node]), model.nodes)

    def test_eater_nodes_getter(self):
        """
        marble_eater_nodes() getter should return only and all MarbleEaterNodes,
        in order of output.
        """
        marble = generate_dummy_marble()
        node = generate_dummy_node()
        eater_node_1 = MarbleEaterNode(
            ZERO, ZERO, ZERO, 0, None, 0, 0, 0, 0, 0)
        eater_node_2 = MarbleEaterNode(
            ZERO, ZERO, ZERO, 0, None, 0, 0, 0, 0, 0)

        nodes = set([node, eater_node_1, eater_node_2])
        model = NenwinModel(nodes, set([marble]))
        expected = list(nodes.difference([node]))
        self.assertListEqual(expected, model.marble_eater_nodes)

    def test_add_marbles(self):
        """
        Base case: add two marbles, should appear in marbles() getter.
        Old marbles should remain as well.
        """
        marble = generate_dummy_marble()
        model = NenwinModel([], set([marble]))

        new_marbles = (generate_dummy_marble(), generate_dummy_marble())

        model.add_marbles(new_marbles)

        expected = set([marble, *new_marbles])
        self.assertSetEqual(expected, model.marbles)
        self.assertSetEqual(expected, model._NenwinModel__all_particles)

    def test_add_marbles_empty(self):
        """
        Corner case: add 0 new Marbles.
        """
        marble = generate_dummy_marble()
        model = NenwinModel([], set([marble]))

        new_marbles = []

        model.add_marbles(new_marbles)

        expected = set([marble])
        self.assertSetEqual(expected, model.marbles)

    def test_make_timestep_1(self):
        """
        Base base: single moving Marble, with constant velocity.
        """
        marble = Marble(ZERO, np.array([10]), ZERO, 1, None, None, 0, 0, 0, 0)
        model = NenwinModel([], [marble])

        model.make_timestep(time_passed=1)
        expected_new_pos = np.array([10])

        self.assertTrue(check_close(marble.pos, expected_new_pos))

    def test_make_timestep_2(self):
        """
        Base case: initial stationairy Marble, gets attracted and accelerated.
        """
        node = Node(pos=ZERO,
                    vel=ZERO,
                    acc=ZERO,
                    mass=1,
                    attraction_function=ATTRACT_FUNCT,
                    marble_stiffness=1,
                    node_stiffness=1,
                    marble_attraction=1,
                    node_attraction=0)
        marble = Marble(pos=np.array([5]),
                        vel=ZERO,
                        acc=ZERO,
                        mass=1,
                        attraction_function=ATTRACT_FUNCT,
                        datum=None)

        model = NenwinModel([node], [marble])
        time_passed = 1

        expected_pos, expected_vel = runge_kutta_4_step(marble.pos,
                                                        marble.vel,
                                                        -ATTRACT_FUNCT.value,
                                                        duration=time_passed)
        model.make_timestep(time_passed)

        self.assertTrue(check_close(marble.pos, expected_pos, atol=0.01))
        self.assertTrue(check_close(marble.vel, expected_vel, atol=0.01))
        self.assertTrue(check_close(marble.acc, -ATTRACT_FUNCT.value,
                                    atol=0.01))

        self.assertTrue(check_close(node.pos, ZERO))
        self.assertTrue(check_close(node.vel, ZERO))
        self.assertTrue(check_close(node.acc, ZERO))

    def test_no_initial_marbles(self):
        """
        In vitro test: should not raise error if no Marbles provided.
        """
        try:
            model = NenwinModel([])
        except:
            self.fail("Initialization of NenwinModel without initial_marbles"
                      + " should not fail.")


def generate_dummy_marble() -> Marble:
    """
    Create a Marble without meaningfull parameters.
    """
    return Marble(ZERO, ZERO, ZERO, 0, None, None, 0, 0, 0, 0)


def generate_dummy_node() -> Marble:
    """
    Create a Node without meaningfull parameters.
    """
    return Node(ZERO, ZERO, ZERO, 0, None, 0, 0, 0, 0)


if __name__ == '__main__':
    unittest.main()
