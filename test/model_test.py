"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Unit-tests for model.py.
"""
import unittest
import numpy as np
from typing import Tuple

from experiment_1.node import Node
from experiment_1.marble import Marble
from experiment_1.model import NenwinModel
from experiment_1.attraction_functions.attraction_functions \
    import AttractionFunction
from experiment_1.particle import PhysicalParticle
from test_aux import check_close
from test_aux import TEST_SIMULATION_STEP_SIZE
from test_aux import runge_kutta_4_step


class TestAttractionFunction(AttractionFunction):
    """
    Simplistic constant function to ease testing numerically.
    (For constant forces => constant acceleration,
    and I can find a precise value for this)
    """

    def compute_attraction(self,
                           first_particle: PhysicalParticle,
                           second_particle: PhysicalParticle
                           ) -> float:
        return 0.01


class ModelTestCase(unittest.TestCase):

    def test_model_run_1(self):
        """
        Base case: single marble and single node.
        """
        attract_funct = TestAttractionFunction()
        zero_vector = np.array([0, 0])

        node = Node(pos=zero_vector, vel=zero_vector, acc=zero_vector,
                    mass=2, attraction_function=attract_funct, stiffness=1)
        marble = Marble(pos=np.array([10, 0]), vel=zero_vector, acc=zero_vector,
                        mass=1, attraction_function=attract_funct, datum=None)

        model = NenwinModel([node],
                            TEST_SIMULATION_STEP_SIZE,
                            [marble])
        # Simulate 1 unit time
        num_steps = (1/TEST_SIMULATION_STEP_SIZE)
        model.run(max_num_steps=num_steps)

        # The node should not have changed at all
        self.assertTrue(check_close(node.acc, zero_vector))
        self.assertTrue(check_close(node.vel, zero_vector))
        self.assertTrue(check_close(node.pos, zero_vector))

        # The marble should have been pulled towards the node
        difference = node.pos - marble.pos
        direction = difference / np.linalg.norm(difference)
        expected_acc = direction * attract_funct(node, marble) / marble.mass
        expected_pos, expected_vel = runge_kutta_4_step(
            pos=np.array([10, 0]), vel=zero_vector, acc=expected_acc,
            duration=1)
        self.assertTrue(check_close(marble.acc, expected_acc))
        self.assertTrue(check_close(marble.vel, expected_vel))
        self.assertTrue(check_close(marble.pos, expected_pos))

    def test_model_run_2(self):
        """
        Corner case: marble stationairy in equilibrium in middle of 2 nodes.
        """
        attract_funct = TestAttractionFunction()
        zero_vector = np.array([0, 0])

        node1 = Node(pos=zero_vector, vel=zero_vector, acc=zero_vector,
                     mass=2, attraction_function=attract_funct, stiffness=1)
        node2 = Node(pos=np.array([2, 2]), vel=zero_vector, acc=zero_vector,
                     mass=2, attraction_function=attract_funct, stiffness=1)
        marble = Marble(pos=np.array([1, 1]), vel=zero_vector, acc=zero_vector,
                        mass=1, attraction_function=attract_funct, datum=None)

        model = NenwinModel([node1, node2],
                            TEST_SIMULATION_STEP_SIZE,
                            [marble])
        model.run(max_num_steps=1)

        # No particle should have changed at all
        self.assertTrue(check_close(node1.acc, zero_vector))
        self.assertTrue(check_close(node1.vel, zero_vector))
        self.assertTrue(check_close(node1.pos, zero_vector))

        self.assertTrue(check_close(node2.acc, zero_vector))
        self.assertTrue(check_close(node2.vel, zero_vector))
        self.assertTrue(check_close(node2.pos, np.array([2, 2])))

        self.assertTrue(check_close(marble.acc, zero_vector))
        self.assertTrue(check_close(marble.vel, zero_vector))
        self.assertTrue(check_close(marble.pos, np.array([1, 1])))

    def test_model_run_3(self):
        """
        Corner case: marble in equilibrium in middle of 2 nodes,
        but with constant velocity in orthogonal direction.
        """
        attract_funct = TestAttractionFunction()
        zero_vector = np.array([0, 0])

        node1 = Node(pos=zero_vector, vel=zero_vector, acc=zero_vector,
                     mass=2, attraction_function=attract_funct, stiffness=1)
        node2 = Node(pos=np.array([2, 0]), vel=zero_vector, acc=zero_vector,
                     mass=2, attraction_function=attract_funct, stiffness=1)
        marble = Marble(pos=np.array([1, 0]), vel=np.array([1, 0]),
                        acc=zero_vector, mass=1,
                        attraction_function=attract_funct,
                        datum=None)

        model = NenwinModel([node1, node2],
                            TEST_SIMULATION_STEP_SIZE,
                            [marble])
        model.run(max_num_steps=1)

        self.assertTrue(check_close(node1.acc, zero_vector))
        self.assertTrue(check_close(node1.vel, zero_vector))
        self.assertTrue(check_close(node1.pos, zero_vector))

        self.assertTrue(check_close(node2.acc, zero_vector))
        self.assertTrue(check_close(node2.vel, zero_vector))
        self.assertTrue(check_close(node2.pos, np.array([2, 0])))

        expected_pos = np.array([1, 0]) \
            + TEST_SIMULATION_STEP_SIZE*np.array([1, 0])
        self.assertTrue(check_close(marble.acc, zero_vector))
        self.assertTrue(check_close(marble.vel, np.array([1, 0])))
        self.assertTrue(check_close(marble.pos, expected_pos))

    def test_model_run_4(self):
        """
        Base case: marble attracted by 2 nodes in non-orthogonal directions.
        """
        attract_funct = TestAttractionFunction()
        zero_vector = np.array([0, 0])

        node1 = Node(pos=np.array([0, 1]),
                     vel=zero_vector,
                     acc=zero_vector,
                     mass=2,
                     attraction_function=attract_funct,
                     stiffness=1)
        node2 = Node(pos=np.array([0, -1]),
                     vel=zero_vector, 
                     acc=zero_vector,
                     mass=2,
                     attraction_function=attract_funct,
                     stiffness=1)
        marble = Marble(pos=np.array([1, 0]),
                        vel=zero_vector,
                        acc=zero_vector,
                        mass=1,
                        attraction_function=attract_funct,
                        datum=None)

        model = NenwinModel([node1, node2],
                            TEST_SIMULATION_STEP_SIZE,
                            [marble])
        model.run(max_num_steps=1)

        self.assertTrue(check_close(node1.acc, zero_vector))
        self.assertTrue(check_close(node1.vel, zero_vector))
        self.assertTrue(check_close(node1.pos, np.array([0, 1])))

        self.assertTrue(check_close(node2.acc, zero_vector))
        self.assertTrue(check_close(node2.vel, zero_vector))
        self.assertTrue(check_close(node2.pos, np.array([0, -1])))

        difference1 = node1.pos - marble.pos
        direction1 = difference1 / np.linalg.norm(difference1)
        difference2 = node2.pos - marble.pos
        direction2 = difference2 / np.linalg.norm(difference2)
        # Newton's Second Law:
        expected_acc = (direction1 * attract_funct(node1, marble) \
            + direction2 * attract_funct(node2, marble))/marble.mass
        expected_pos, expected_vel = runge_kutta_4_step(
            pos=np.array([1, 0]), vel=zero_vector, acc=expected_acc,
            duration=TEST_SIMULATION_STEP_SIZE)
        self.assertTrue(check_close(marble.acc, expected_acc))
        self.assertTrue(check_close(marble.vel, expected_vel))
        self.assertTrue(check_close(marble.pos, expected_pos))

    def test_model_run_4(self):
        """
        Base case: single marble and a single movable node.
        (stiffness != 1)
        """
        attract_funct = TestAttractionFunction()
        zero_vector = np.array([0, 0])
        stiffness = 0.5

        node = Node(pos=zero_vector,
                    vel=zero_vector,
                    acc=zero_vector,
                    mass=2,
                    attraction_function=attract_funct,
                    stiffness=stiffness)
        marble = Marble(pos=np.array([1, 0]),
                        vel=zero_vector,
                        acc=zero_vector,
                        mass=1,
                        attraction_function=attract_funct,
                        datum=None)

        model = NenwinModel([node],
                            TEST_SIMULATION_STEP_SIZE,
                            [marble])
        # Simulate 1 unit time
        num_steps = (1/TEST_SIMULATION_STEP_SIZE)
        model.run(max_num_steps=num_steps)

        difference = marble.pos - node.pos
        direction_node = difference / np.linalg.norm(difference)
        expected_acc_node = stiffness * direction_node \
            * attract_funct(node, marble) / node.mass #TODO: removing node.mass here creates very different result!
        self.assertEqual(node.mass, 2)
        expected_pos_node, expected_vel_node = runge_kutta_4_step(
            pos=zero_vector, vel=zero_vector, acc=expected_acc_node,
            duration=1)
        self.assertTrue(check_close(node.acc, expected_acc_node))
        self.assertTrue(check_close(node.vel, expected_vel_node))
        self.assertTrue(check_close(node.pos, expected_pos_node))

        difference = -difference
        direction_marble = difference / np.linalg.norm(difference)
        expected_acc_marble = direction_marble \
            * attract_funct(node, marble)
        expected_pos_marble, expected_vel_marble = runge_kutta_4_step(
            pos=np.array([1, 0]), vel=zero_vector, acc=expected_acc_marble,
            duration=1)
        self.assertTrue(check_close(marble.acc, expected_acc_marble))
        self.assertTrue(check_close(marble.vel, expected_vel_marble))
        self.assertTrue(check_close(marble.pos, expected_pos_marble))


if __name__ == '__main__':
    unittest.main()
