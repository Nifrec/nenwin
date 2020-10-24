"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Unit-tests for node.py.
"""
import unittest
import numpy as np
from typing import Tuple

from experiment_1.node import Node, MarbleEaterNode
from experiment_1.marble import Marble
from experiment_1.attraction_functions.attraction_functions import Gratan
from test_aux import check_close


class NodeTestCase(unittest.TestCase):

    def test_stiffness_property_1(self):
        """
        Base case: just test the getter.
        """
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([10, 9, 0])
        mass = 1
        stiffness = 0.5
        node = Node(pos, vel, acc, mass, lambda: None, stiffness)

        self.assertEqual(node.stiffness, stiffness)

    def test_stiffness_property_2(self):
        """
        Corner case: invalid value, too high
        """
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([10, 9, 0])
        mass = 1
        stiffness = 1.0001
        args = (pos, vel, acc, mass, lambda: None, stiffness)

        self.assertRaises(ValueError, Node, *args)

    def test_stiffness_property_3(self):
        """
        Corner case: invalid value, too low
        """
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([10, 9, 0])
        mass = 1
        stiffness = -0.0001
        args = (pos, vel, acc, mass, lambda: None, stiffness)

        self.assertRaises(ValueError, Node, *args)

    def test_stiffness_acc_update_1(self):
        """
        Base case: non-boundary stiffness.
        """
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        mass = 1
        stiffness = 0.5
        node = Node(pos, vel, acc, mass, lambda: None, stiffness)

        forces = np.array([[1, 1, 1]])
        node.update_acceleration(forces)

        expected = np.array([1, 1, 1]) * stiffness

        self.assertTrue(check_close(node.acc, expected))

    def test_stiffness_acc_update_2(self):
        """
        Corner case: 0 stiffness.
        """
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        mass = 1
        stiffness = 0
        node = Node(pos, vel, acc, mass, lambda: None, stiffness)

        forces = np.array([[1, 1, 1]])
        node.update_acceleration(forces)

        expected = np.array([1, 1, 1])

        self.assertTrue(check_close(node.acc, expected))

    def test_stiffness_acc_update_3(self):
        """
        Corner case: 0 stiffness.
        """
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        mass = 1
        stiffness = 1
        node = Node(pos, vel, acc, mass, lambda: None, stiffness)

        forces = np.array([[1, 1, 1]])
        node.update_acceleration(forces)

        self.assertTrue(check_close(node.acc, acc))


class MarbleEaterNodeTestCase(unittest.TestCase):

    def test_eat_1(self):
        """
        Base case: eat a Marble, test if stored correctly.
        """
        # Irrelevant parameters
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([10, 9, 0])
        mass = 1
        stiffness = 0.5
        radius = 0

        node = MarbleEaterNode(pos, vel, acc, mass, lambda: None,
                               stiffness, radius)
        datum = "Hell0 W0rld!"
        marble = Marble(pos, vel, acc, mass, lambda: None, datum)

        node.eat(marble)
        del marble # It is supposed to dissapear now in the Simulation as well.

        self.assertEqual(node.num_marbles_eaten, 1)
        self.assertListEqual(node.marble_data_eaten, [datum])

    def test_eat_2(self):
        """
        Corner case: no Marbles eaten.
        """
        # Irrelevant parameters
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([10, 9, 0])
        mass = 1
        stiffness = 0.5
        radius = 0

        node = MarbleEaterNode(pos, vel, acc, mass, lambda: None,
                               stiffness, radius)

        self.assertEqual(node.num_marbles_eaten, 0)
        self.assertListEqual(node.marble_data_eaten, [])

    def test_eat_2(self):
        """
        Base case: eat 2 Marbles, test if stored correctly in right order.
        """
        # Irrelevant parameters
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([10, 9, 0])
        mass = 1
        stiffness = 0.5
        radius = 0

        node = MarbleEaterNode(pos, vel, acc, mass, lambda: None,
                               stiffness, radius)
        datum1 = "Hell0 W0rld!"
        marble1 = Marble(pos, vel, acc, mass, lambda: None, datum1)

        node.eat(marble1)
        del marble1 # It is supposed to dissapear now in the Simulation as well.

        datum2 = set([1, 2, 3])
        marble2 = Marble(pos, vel, acc, mass, lambda: None, datum2)
        node.eat(marble2)
        del marble2

        self.assertEqual(node.num_marbles_eaten, 2)
        self.assertListEqual(node.marble_data_eaten, [datum1, datum2])






if __name__ == '__main__':
    unittest.main()
