"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
March 2021

Copyright (C) 2021 Lulof Pirée, Teun Schilperoort

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

Unit-tests for NenwinModel class of model.py.
"""
import unittest
import numpy as np
import torch
from typing import Tuple

from nenwin.node import Node, Marble
from nenwin.model import NenwinModel
from nenwin.attraction_functions.attraction_functions \
    import AttractionFunction
from nenwin.particle import PhysicalParticle
from nenwin.marble_eater_node import MarbleEaterNode
from nenwin.test.test_aux import TEST_SIMULATION_STEP_SIZE
from nenwin.test.test_aux import runge_kutta_4_step
from nenwin.test.test_aux import ATTRACT_FUNCT
from nenwin.test.test_aux import ZERO
from nenwin.attraction_functions.attraction_functions import NewtonianGravity


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
        expected_new_pos = torch.tensor([10], dtype=torch.float)

        self.assertTrue(torch.allclose(marble.pos, expected_new_pos))

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

        expected_pos, expected_vel = runge_kutta_4_step(
            marble.pos,
            marble.vel,
            -ATTRACT_FUNCT.value,
            duration=time_passed)
        model.make_timestep(time_passed)

        self.assertTrue(torch.allclose(marble.pos, expected_pos, atol=0.01))
        self.assertTrue(torch.allclose(marble.vel, expected_vel, atol=0.01))
        self.assertTrue(torch.isclose(marble.acc,
                                      torch.tensor(-ATTRACT_FUNCT.value),
                                      atol=0.01))

        self.assertTrue(torch.allclose(node.pos, ZERO))
        self.assertTrue(torch.allclose(node.vel, ZERO))
        self.assertTrue(torch.allclose(node.acc, ZERO))

    def test_no_initial_marbles(self):
        """
        In vitro test: should not raise error if no Marbles provided.
        """
        try:
            model = NenwinModel([])
        except:
            self.fail("Initialization of NenwinModel without initial_marbles"
                      + " should not fail.")


class ModelBackpropTestCase(unittest.TestCase):
    """
    Testcase tests if backpropagation correctly behaves for particle
    interaction within a model.
    """

    def setUp(self):
        self.node = Node(pos=ZERO,
                         vel=ZERO,
                         acc=ZERO,
                         mass=1,
                         attraction_function=NewtonianGravity(),
                         marble_stiffness=1,
                         node_stiffness=1,
                         marble_attraction=1,
                         node_attraction=0)
        self.marble = Marble(pos=np.array([5]),
                             vel=ZERO,
                             acc=ZERO,
                             mass=1,
                             attraction_function=ATTRACT_FUNCT,
                             datum=None)

        self.model = NenwinModel([self.node], [self.marble])

    def test_particle_gradients(self):
        """
        Base case: a Marble being attracted by a stationary Node,
        when using Marble's variables to compute loss,
        also Node.pos should receive gradients 
        when computing backprop on the loss.
        """
        self.model.make_timestep(1.0)
        self.model.make_timestep(1.0)
        loss = 2 * self.marble.pos
        loss.backward()

        self.assertIsNotNone(self.node.init_pos.grad)

    def test_particle_gradients_moving_node(self):
        """
        Base case: Marble being attracted by a moving Node,
        when using Marble's variables to compute loss,
        also Node.pos should receive gradients 
        when computing backprop on the loss.
        """
        self.node = Node(pos=ZERO,
                         vel=torch.tensor([0.1]),
                         acc=ZERO,
                         mass=1,
                         attraction_function=NewtonianGravity(),
                         marble_stiffness=1,
                         node_stiffness=1,
                         marble_attraction=1,
                         node_attraction=0)
        self.model = NenwinModel([self.node], [self.marble])
        self.model.make_timestep(1.0)
        self.model.make_timestep(1.0)
        self.model.make_timestep(1.0)

        loss = 2 * self.marble.acc
        loss.backward()

        self.assertIsNotNone(self.marble.init_pos.grad)
        self.assertIsNotNone(self.node.init_pos.grad)
        self.assertIsNotNone(self.node.init_vel.grad)
        self.assertIsNotNone(self.node._PhysicalParticle__mass.grad)


class ModelToStringTestCase(unittest.TestCase):
    """
    Test if a NenwinModel (including stored particles)
    can successfully be converted to a string,
    and also be reloaded from a string.
    """

    def setUp(self):
        self.maxDiff = None

    def test_repr_1(self):
        """
        Base case: no Nodes, empty set initial Marbles given.
        """
        expected = "NenwinModel(set(),set())"
        result = repr(NenwinModel([], []))
        self.assertEqual(expected, result)

    def test_repr_2(self):
        """
        Base case: no Nodes, no set initial Marbles given.
        """
        expected = "NenwinModel(set(),set())"
        result = repr(NenwinModel([]))
        self.assertEqual(expected, result)

    def test_repr_3(self):
        """
        Base case: Nodes and Marbles given.
        """
        marble_pos = torch.tensor([1.])
        marble_vel = torch.tensor([2.])
        marble_acc = torch.tensor([3.])
        marble = Marble(marble_pos, marble_vel, marble_acc, 8,
                        None, None, 0.4, 0.5, 0.6, 0.7)
        marble_2 = marble.copy()
        node = Node(ZERO, ZERO, ZERO, 0, None, 0, 0, 0, 0)
        node_2 = Node(torch.tensor([11.]), torch.tensor([12.]),
                      torch.tensor([13.]), 14, None, 0, 0, 0, 0)
        expected = f"NenwinModel({repr(set([node, node_2]))}," \
            + f"{repr(set([marble, marble_2]))})"
        result = repr(NenwinModel([node, node_2], [marble, marble_2]))
        self.assertEqual(expected, result)

    def test_get_params(self):
        """
        Nodes and Marbles should be registered as submodules
        of the model within the PyTorch framework,
        and hence all learnable parameters should be
        obtainable via model.
        """
        marbles = (generate_dummy_marble(), )
        nodes = (generate_dummy_node(), )
        model = NenwinModel(nodes, marbles)
        # 8 parameters per particle: init_pos, init_vel, init_acc, mass,
        # and the 4 stiffness / attraction params
        result = tuple(model.parameters())
        print(result)
        self.assertEqual(len(result), 16)

    def test_get_params_add_marbles(self):
        """
        Also later added marbles should be registered.
        """
        marbles = (generate_dummy_marble(), )
        nodes = (generate_dummy_node(), )
        model = NenwinModel(nodes)

        model.add_marbles(marbles)

        # 8 parameters per particle: init_pos, init_vel, init_acc, mass,
        # and the 4 stiffness / attraction params
        result = tuple(model.parameters())
        print(result)
        self.assertEqual(len(result), 16)



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
