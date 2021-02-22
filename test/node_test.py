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

Unit-tests for Node of node.py.
"""
import unittest
import numpy as np
import torch

from experiment_1.attraction_functions.attraction_functions import Gratan
from experiment_1.attraction_functions.attraction_functions \
    import NewtonianGravity
from experiment_1.node import Node, Marble
from experiment_1.particle import PhysicalParticle
from experiment_1.auxliary import generate_stiffness_dict
from test_aux import ATTRACT_FUNCT
from test_aux import NUMERICAL_ABS_ACCURACY_REQUIRED
from test_aux import ZERO
from test_aux import check_named_parameters
from test_aux import convert_scalar_param_to_repr


class NodeTestCase(unittest.TestCase):

    def test_stiffness_getters(self):
        marble_stiffness = 0.42
        node_stiffness = 0.13
        particle = create_particle(
            marble_stiffness, node_stiffness, ZERO, ZERO)
        self.assertAlmostEqual(particle.marble_stiffness, marble_stiffness)
        self.assertAlmostEqual(particle.node_stiffness, node_stiffness)

    def test_attraction_getters(self):
        marble_attraction = 0.42
        node_attraction = 0.13
        particle = create_particle(ZERO, ZERO,
                                   marble_attraction, node_attraction)
        self.assertAlmostEqual(particle.marble_attraction, marble_attraction)
        self.assertAlmostEqual(particle.node_attraction, node_attraction)

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
        node = Node(torch.tensor([-1.], dtype=torch.float),
                    ZERO, ZERO, 0, None, 0, 0, 0, 0)
        particle = create_particle(0, 0, 0, node_attraction)

        result = particle.compute_attraction_force_to(node)
        self.assertEqual(result, ATTRACT_FUNCT.value * node_attraction)

    def test_attraction_node_2(self):
        """
        Corner case: should not attract any node if the multiplier is 0.
        """
        node_attraction = 0
        node = Node(torch.tensor([-1.], dtype=torch.float),
                    ZERO, ZERO, 0, None, 0, 0, 1, 1)
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
        marble = Marble(torch.tensor(
            [-1.], dtype=torch.float), ZERO, ZERO, 0, None, None)
        particle = create_particle(0, 0, marble_attraction, 0)

        result = particle.compute_attraction_force_to(marble)
        self.assertEqual(result, ATTRACT_FUNCT.value * marble_attraction)

    def test_attraction_marble_2(self):
        """
        Corner case: should not attract any node if the multiplier is 0.
        """
        marble_attraction = 0
        marble = Marble(torch.tensor(
            [-1.], dtype=torch.float), ZERO, ZERO, 0, None, None)
        particle = create_particle(0, 0, marble_attraction, 0)
        result = particle.compute_attraction_force_to(marble)
        self.assertEqual(result, 0)

    def test_attraction_wrong_type(self):
        """
        Corner case: can only compute attraction to Marbles and Nodes.
        """
        other = PhysicalParticle(torch.tensor(
            [-1.], dtype=torch.float), ZERO, ZERO, 0, None)
        particle = create_particle(0, 0, 0, 0)
        self.assertRaises(
            ValueError, particle.compute_attraction_force_to, other)

    def test_stiffness_to_marble(self):
        """
        Base case: 50% stiffness to a single Marble.
        """
        marble_stiffness = 0.5
        marble = Marble(torch.tensor(
            [1.], dtype=torch.float), ZERO, ZERO, 0, ATTRACT_FUNCT, None)
        particle = create_particle(marble_stiffness, 0, 0, 0)

        expected = (1-marble_stiffness)*ATTRACT_FUNCT.value
        result = particle.compute_experienced_force(set([marble]))
        self.assertEqual(expected, result,
                         "marble_stiffness, "
                         + f"got: {result}, exptected:{expected}")

    def test_stiffness_to_node(self):
        """
        Base case: 1% stiffness to a single Node.
        """
        node_stiffness = 0.01
        node = Node(torch.tensor([1.], dtype=torch.float),
                    ZERO, ZERO, 0, ATTRACT_FUNCT, 0, 0, 1, 1)
        particle = create_particle(0, node_stiffness, 0, 0)

        expected = (1-node_stiffness)*ATTRACT_FUNCT.value
        result = particle.compute_experienced_force(set([node]))
        self.assertEqual(expected, result,
                         "node_stiffness, "
                         + f"got: {result}, exptected:{expected}")

    def test_stiffness_zero(self):
        """
        Corner case: 100% stiffness to any particle.
        """
        node_stiffness = 1
        marble_stiffness = 1
        marble = Marble(torch.tensor(
            [1.], dtype=torch.float), ZERO, ZERO, 0, ATTRACT_FUNCT, None)
        node = Node(torch.tensor([1.], dtype=torch.float),
                    ZERO, ZERO, 0, ATTRACT_FUNCT, 0, 0, 1, 1)
        particle = create_particle(marble_stiffness, node_stiffness, 0, 0)

        expected = ZERO
        result = particle.compute_experienced_force(set([node]))
        self.assertTrue(torch.allclose(expected, result),
                        "zero stiffness, "
                        + f"got: {result}, exptected:{expected}")

    def test_stiffness_error(self):
        """
        Corner case: raise error if one of particles is neither Node nor Marble.
        """
        node_stiffness = 1
        marble_stiffness = 1
        other = PhysicalParticle(ZERO, ZERO, ZERO, 0, ATTRACT_FUNCT)
        particle = create_particle(marble_stiffness, node_stiffness, 0, 0)

        self.assertRaises(ValueError,
                          particle.compute_experienced_force,
                          set([other]))

    def test_stiffness_to_set(self):
        """
        Base case: multiple other particles in input set of 
            compute_experienced_force()
        """
        node_stiffness = 0.1
        marble_stiffness = 0.6
        node = Node(torch.tensor([1.], dtype=torch.float),
                    ZERO, ZERO, 0, ATTRACT_FUNCT, 0, 0, 1, 1)
        marble = Marble(torch.tensor(
            [-1.], dtype=torch.float), ZERO, ZERO, 0, ATTRACT_FUNCT, None, 0, 0, 1, 1)
        particle = create_particle(marble_stiffness, node_stiffness, 0, 0)

        expected = (1 - node_stiffness)*ATTRACT_FUNCT.value \
            - (1 - marble_stiffness)*ATTRACT_FUNCT.value
        result = particle.compute_experienced_force(set([node, marble]))

        self.assertAlmostEqual(expected, result.item(), delta=1e-5)

    def test_copy_1(self):
        """
        Basic test: the copy should have similarly valued 
        attribute values as the original.
        """
        pos = torch.tensor([1.], dtype=torch.float)
        vel = torch.tensor([2.], dtype=torch.float)
        acc = torch.tensor([3.], dtype=torch.float)
        mass = 4
        attraction_funct = ATTRACT_FUNCT
        stiffnesses = generate_stiffness_dict(0.5, 0.6, 0.7, 0.8)
        original = Node(pos, vel, acc, mass, attraction_funct,
                        **stiffnesses)
        copy = original.copy()

        self.assertFalse(copy is original)

        self.assertTrue(torch.allclose(acc, copy.acc))
        self.assertTrue(torch.allclose(vel, copy.vel))
        self.assertTrue(torch.allclose(pos, copy.pos))
        self.assertEqual(mass, copy.mass)
        self.assertTrue(attraction_funct is copy._attraction_function)
        self.assertAlmostEqual(copy.marble_stiffness,
                               stiffnesses["marble_stiffness"])
        self.assertAlmostEqual(copy.node_stiffness,
                               stiffnesses["node_stiffness"])
        self.assertAlmostEqual(copy.marble_attraction,
                               stiffnesses["marble_attraction"])
        self.assertAlmostEqual(copy.node_attraction,
                               stiffnesses["node_attraction"])

    def test_copy_2(self):
        """
        Implementation test: the copy should have exactly the same
        learnable parameters as the original. 
        This ensures backpropagation will propagate from the copies to the
        original.
        """
        pass


    def test_parameters(self):
        pos = torch.tensor([1], dtype=torch.float)
        vel = torch.tensor([2], dtype=torch.float)
        acc = torch.tensor([3], dtype=torch.float)
        mass = 4
        attraction_funct = ATTRACT_FUNCT
        stiffnesses = generate_stiffness_dict(0.5, 0.6, 0.7, 0.8)
        node = Node(pos, vel, acc, mass, attraction_funct,
                    **stiffnesses)

        named_params = node.named_parameters()
        expected_names = {
            '_Node__marble_stiffness': stiffnesses["marble_stiffness"],
            '_Node__node_stiffness': stiffnesses["node_stiffness"],
            '_Node__marble_attraction': stiffnesses["marble_attraction"],
            '_Node__node_attraction': stiffnesses["node_attraction"]}
        self.assertTrue(check_named_parameters(expected_names,
                                               tuple(named_params)))

    def test_repr(self):
        pos = torch.tensor([0], dtype=torch.float)
        vel = torch.tensor([1], dtype=torch.float)
        acc = torch.tensor([2], dtype=torch.float)
        mass = 3.0
        attraction_function = NewtonianGravity()
        marble_stiffness = 0.4
        node_stiffness = 0.5
        marble_attraction = 0.6
        node_attraction = 0.7

        node = Node(pos, vel, acc, mass, attraction_function, marble_stiffness,
                    node_stiffness, marble_attraction, node_attraction)

        # Some numerical errors occurs when converting from float to FloatTensor
        marble_stiffness_float_repr = \
            convert_scalar_param_to_repr(marble_stiffness)
        node_stiffness_float_repr = convert_scalar_param_to_repr(node_stiffness)
        marble_attraction_float_repr = \
            convert_scalar_param_to_repr(marble_attraction)
        node_attraction_float_repr = \
            convert_scalar_param_to_repr(node_attraction)
        expected = f"Node({repr(pos)},{repr(vel)},"\
            + f"{repr(acc)},{mass},NewtonianGravity(),"\
            + f"{marble_stiffness_float_repr}," \
            + f"{node_stiffness_float_repr},{marble_attraction_float_repr}," \
            + f"{node_attraction_float_repr})"
        result = repr(node)
        self.assertEqual(expected, result)


def create_particle(marble_stiffness,
                    node_stiffness,
                    marble_attraction,
                    node_attraction) -> Node:
    """
    Simply attempt to create a Node with given parameters,
    and 0 or None for all other parameter values.
    """
    return Node(ZERO, ZERO, ZERO, 0, ATTRACT_FUNCT,
                marble_stiffness=marble_stiffness,
                node_stiffness=node_stiffness,
                marble_attraction=marble_attraction,
                node_attraction=node_attraction)


if __name__ == '__main__':
    unittest.main()
