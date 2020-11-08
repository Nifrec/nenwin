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

from experiment_1.stiffness_particle import Node, Marble
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
        Base base: single moving Marble.
        """
        marble = Marble(ZERO, np.array([10]), ZERO, 1, None, None, 0, 0, 0, 0)
        model = NenwinModel([], [marble])

        model.make_timestep(time_passed=1)
        expected_new_pos = np.array([10])

        self.assertTrue(check_close(marble.pos, expected_new_pos))

    def test_make_timestep_2(self):
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
        self.assertTrue(check_close(marble.acc, -ATTRACT_FUNCT.value, atol=0.01))

    # def test_model_run_1(self):
    #     """
    #     Base case: single marble and single node.
    #     """
    #     attract_funct = TestAttractionFunction()
    #     zero_vector = np.array([0, 0])

    #     node = Node(pos=zero_vector, vel=zero_vector, acc=zero_vector,
    #                 mass=2, attraction_function=attract_funct, stiffness=1)
    #     marble = Marble(pos=np.array([10, 0]), vel=zero_vector, acc=zero_vector,
    #                     mass=1, attraction_function=attract_funct, datum=None)

    #     model = NenwinModel([node],
    #                         TEST_SIMULATION_STEP_SIZE,
    #                         [marble])
    #     # Simulate 1 unit time
    #     num_steps = (1/TEST_SIMULATION_STEP_SIZE)
    #     model.run(max_num_steps=num_steps)

    #     # The node should not have changed at all
    #     self.assertTrue(check_close(node.acc, zero_vector))
    #     self.assertTrue(check_close(node.vel, zero_vector))
    #     self.assertTrue(check_close(node.pos, zero_vector))

    #     # The marble should have been pulled towards the node
    #     difference = node.pos - marble.pos
    #     direction = difference / np.linalg.norm(difference)
    #     expected_acc = direction * attract_funct(node, marble) / marble.mass
    #     expected_pos, expected_vel = runge_kutta_4_step(
    #         pos=np.array([10, 0]), vel=zero_vector, acc=expected_acc,
    #         duration=1)
    #     self.assertTrue(check_close(marble.acc, expected_acc))
    #     self.assertTrue(check_close(marble.vel, expected_vel))
    #     self.assertTrue(check_close(marble.pos, expected_pos))

    # def test_model_run_2(self):
    #     """
    #     Corner case: marble stationairy in equilibrium in middle of 2 nodes.
    #     """
    #     attract_funct = TestAttractionFunction()
    #     zero_vector = np.array([0, 0])

    #     node1 = Node(pos=zero_vector, vel=zero_vector, acc=zero_vector,
    #                  mass=2, attraction_function=attract_funct, stiffness=1)
    #     node2 = Node(pos=np.array([2, 2]), vel=zero_vector, acc=zero_vector,
    #                  mass=2, attraction_function=attract_funct, stiffness=1)
    #     marble = Marble(pos=np.array([1, 1]), vel=zero_vector, acc=zero_vector,
    #                     mass=1, attraction_function=attract_funct, datum=None)

    #     model = NenwinModel([node1, node2],
    #                         TEST_SIMULATION_STEP_SIZE,
    #                         [marble])
    #     model.run(max_num_steps=1)

    #     # No particle should have changed at all
    #     self.assertTrue(check_close(node1.acc, zero_vector))
    #     self.assertTrue(check_close(node1.vel, zero_vector))
    #     self.assertTrue(check_close(node1.pos, zero_vector))

    #     self.assertTrue(check_close(node2.acc, zero_vector))
    #     self.assertTrue(check_close(node2.vel, zero_vector))
    #     self.assertTrue(check_close(node2.pos, np.array([2, 2])))

    #     self.assertTrue(check_close(marble.acc, zero_vector))
    #     self.assertTrue(check_close(marble.vel, zero_vector))
    #     self.assertTrue(check_close(marble.pos, np.array([1, 1])))

    # def test_model_run_3(self):
    #     """
    #     Corner case: marble in equilibrium in middle of 2 nodes,
    #     but with constant velocity in orthogonal direction.
    #     """
    #     attract_funct = TestAttractionFunction()
    #     zero_vector = np.array([0, 0])

    #     node1 = Node(pos=zero_vector, vel=zero_vector, acc=zero_vector,
    #                  mass=2, attraction_function=attract_funct, stiffness=1)
    #     node2 = Node(pos=np.array([2, 0]), vel=zero_vector, acc=zero_vector,
    #                  mass=2, attraction_function=attract_funct, stiffness=1)
    #     marble = Marble(pos=np.array([1, 0]), vel=np.array([1, 0]),
    #                     acc=zero_vector, mass=1,
    #                     attraction_function=attract_funct,
    #                     datum=None)

    #     model = NenwinModel([node1, node2],
    #                         TEST_SIMULATION_STEP_SIZE,
    #                         [marble])
    #     model.run(max_num_steps=1)

    #     self.assertTrue(check_close(node1.acc, zero_vector))
    #     self.assertTrue(check_close(node1.vel, zero_vector))
    #     self.assertTrue(check_close(node1.pos, zero_vector))

    #     self.assertTrue(check_close(node2.acc, zero_vector))
    #     self.assertTrue(check_close(node2.vel, zero_vector))
    #     self.assertTrue(check_close(node2.pos, np.array([2, 0])))

    #     expected_pos = np.array([1, 0]) \
    #         + TEST_SIMULATION_STEP_SIZE*np.array([1, 0])
    #     self.assertTrue(check_close(marble.acc, zero_vector))
    #     self.assertTrue(check_close(marble.vel, np.array([1, 0])))
    #     self.assertTrue(check_close(marble.pos, expected_pos))

    # def test_model_run_4(self):
    #     """
    #     Base case: marble attracted by 2 nodes in opposite directions.
    #     """
    #     attract_funct = TestAttractionFunction()
    #     zero_vector = np.array([0, 0])

    #     node1 = Node(pos=np.array([0, 1]),
    #                  vel=zero_vector,
    #                  acc=zero_vector,
    #                  mass=2,
    #                  attraction_function=attract_funct,
    #                  stiffness=1)
    #     node2 = Node(pos=np.array([0, -1]),
    #                  vel=zero_vector,
    #                  acc=zero_vector,
    #                  mass=2,
    #                  attraction_function=attract_funct,
    #                  stiffness=1)
    #     marble = Marble(pos=np.array([1, 0]),
    #                     vel=zero_vector,
    #                     acc=zero_vector,
    #                     mass=1,
    #                     attraction_function=attract_funct,
    #                     datum=None)

    #     model = NenwinModel([node1, node2],
    #                         TEST_SIMULATION_STEP_SIZE,
    #                         [marble])
    #     model.run(max_num_steps=1)

    #     self.assertTrue(check_close(node1.acc, zero_vector))
    #     self.assertTrue(check_close(node1.vel, zero_vector))
    #     self.assertTrue(check_close(node1.pos, np.array([0, 1])))

    #     self.assertTrue(check_close(node2.acc, zero_vector))
    #     self.assertTrue(check_close(node2.vel, zero_vector))
    #     self.assertTrue(check_close(node2.pos, np.array([0, -1])))

    #     difference1 = node1.pos - marble.pos
    #     direction1 = difference1 / np.linalg.norm(difference1)
    #     difference2 = node2.pos - marble.pos
    #     direction2 = difference2 / np.linalg.norm(difference2)
    #     # Newton's Second Law:
    #     expected_acc = np.array([0, 0])
    #     expected_pos, expected_vel = runge_kutta_4_step(
    #         pos=np.array([1, 0]), vel=zero_vector, acc=expected_acc,
    #         duration=TEST_SIMULATION_STEP_SIZE)
    #     self.assertTrue(check_close(marble.acc, expected_acc))
    #     self.assertTrue(check_close(marble.vel, expected_vel))
    #     self.assertTrue(check_close(marble.pos, expected_pos))

    # def test_model_run_5(self):
    #     """
    #     Base case: single marble and a single movable node.
    #     (stiffness != 1)
    #     """
    #     attract_funct = TestAttractionFunction()
    #     zero_vector = np.array([0, 0])
    #     stiffness = 0.5

    #     node = Node(pos=zero_vector,
    #                 vel=zero_vector,
    #                 acc=zero_vector,
    #                 mass=2,
    #                 attraction_function=attract_funct,
    #                 stiffness, stiffness, 1, 1)
    #     marble = Marble(pos=np.array([1, 0]),
    #                     vel=zero_vector,
    #                     acc=zero_vector,
    #                     mass=1,
    #                     attraction_function=attract_funct,
    #                     datum=None)

    #     model = NenwinModel([node],
    #                         TEST_SIMULATION_STEP_SIZE,
    #                         [marble])
    #     # Simulate 1 unit time
    #     num_steps = (1/TEST_SIMULATION_STEP_SIZE)
    #     model.run(max_num_steps=num_steps)

    #     difference = marble.pos - node.pos
    #     direction_node = difference / np.linalg.norm(difference)
    #     expected_acc_node = stiffness * direction_node \
    #         * attract_funct(node, marble) / node.mass #TODO: removing node.mass here creates very different result!
    #     self.assertEqual(node.mass, 2)
    #     expected_pos_node, expected_vel_node = runge_kutta_4_step(
    #         pos=zero_vector, vel=zero_vector, acc=expected_acc_node,
    #         duration=1)
    #     self.assertTrue(check_close(node.acc, expected_acc_node))
    #     self.assertTrue(check_close(node.vel, expected_vel_node))
    #     self.assertTrue(check_close(node.pos, expected_pos_node))

    #     difference = -difference
    #     direction_marble = difference / np.linalg.norm(difference)
    #     expected_acc_marble = direction_marble \
    #         * attract_funct(node, marble)
    #     expected_pos_marble, expected_vel_marble = runge_kutta_4_step(
    #         pos=np.array([1, 0]), vel=zero_vector, acc=expected_acc_marble,
    #         duration=1)
    #     self.assertTrue(check_close(marble.acc, expected_acc_marble))
    #     self.assertTrue(check_close(marble.vel, expected_vel_marble))
    #     self.assertTrue(check_close(marble.pos, expected_pos_marble))


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
