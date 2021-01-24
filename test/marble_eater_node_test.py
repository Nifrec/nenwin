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

Unit-tests for marble_eater_node.py.
"""
import unittest
import torch
import numpy as np
from typing import Tuple

from experiment_1.marble_eater_node import MarbleEaterNode
from experiment_1.node import Marble
from experiment_1.attraction_functions.attraction_functions import Gratan
from experiment_1.attraction_functions.attraction_functions \
    import NewtonianGravity
from experiment_1.auxliary import generate_stiffness_dict
from test_aux import ZERO, ATTRACT_FUNCT
from test_aux import check_named_parameters
from test_aux import convert_scalar_param_to_repr


class MarbleEaterNodeTestCase(unittest.TestCase):

    def test_eat_1(self):
        """
        Base case: eat a Marble, test if stored correctly.
        """
        # Irrelevant parameters
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([10, 9, 0], dtype=torch.float)
        mass = 1
        radius = 0

        node = MarbleEaterNode(pos, vel, acc, mass, lambda: None,
                               0, 0, 0, 0, radius)
        datum = "Hell0 W0rld!"
        marble = Marble(pos, vel, acc, mass, lambda: None, datum)

        node.eat(marble)
        # It is supposed to dissapear now in the Simulation as well.
        del marble

        self.assertEqual(node.num_marbles_eaten, 1)
        self.assertListEqual(node.marble_data_eaten, [datum])

    def test_eat_2(self):
        """
        Corner case: no Marbles eaten.
        """
        # Irrelevant parameters
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([10, 9, 0], dtype=torch.float)
        mass = 1
        stiffness = 0.5
        radius = 0

        node = MarbleEaterNode(pos, vel, acc, mass, lambda: None,
                               0, 0, 0, 0, radius)

        self.assertEqual(node.num_marbles_eaten, 0)
        self.assertListEqual(node.marble_data_eaten, [])

    def test_eat_3(self):
        """
        Base case: eat 2 Marbles, test if stored correctly in right order.
        """
        # Irrelevant parameters
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([10, 9, 0], dtype=torch.float)
        mass = 1
        stiffness = 0.5
        radius = 0

        node = MarbleEaterNode(pos, vel, acc, mass, lambda: None,
                               0, 0, 0, 0, radius)
        datum1 = "Hell0 W0rld!"
        marble1 = Marble(pos, vel, acc, mass, lambda: None, datum1)

        node.eat(marble1)
        # It is supposed to dissapear now in the Simulation as well.
        del marble1

        datum2 = set([1, 2, 3])
        marble2 = Marble(pos, vel, acc, mass, lambda: None, datum2)
        node.eat(marble2)
        del marble2

        self.assertEqual(node.num_marbles_eaten, 2)
        self.assertListEqual(node.marble_data_eaten, [datum1, datum2])

    def test_copy(self):
        pos = torch.tensor([1], dtype=torch.float)
        vel = torch.tensor([2], dtype=torch.float)
        acc = torch.tensor([3], dtype=torch.float)
        mass = 4
        attraction_funct = ATTRACT_FUNCT
        stiffnesses = generate_stiffness_dict(0.5, 0.6, 0.7, 0.8)
        radius = 9
        original = MarbleEaterNode(pos,
                                   vel,
                                   acc,
                                   mass,
                                   attraction_funct,
                                   radius=radius,
                                   ** stiffnesses)
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
        self.assertEqual(copy.radius, radius)
        self.assertEqual(copy.num_marbles_eaten, 0)

    def test_named_parameters(self):

        pos = torch.tensor([1], dtype=torch.float)
        vel = torch.tensor([2], dtype=torch.float)
        acc = torch.tensor([3], dtype=torch.float)
        mass = 4
        attraction_funct = ATTRACT_FUNCT
        stiffnesses = generate_stiffness_dict(0.5, 0.6, 0.7, 0.8)
        radius = 9
        eater = MarbleEaterNode(pos, vel, acc, mass,
                                attraction_funct,
                                radius=radius,
                                ** stiffnesses)

        named_params = eater.named_parameters()
        expected_names = {'_MarbleEaterNode__radius': radius}
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
        radius = 8.0
        # Some numerical errors occurs when converting from float to FloatTensor
        marble_stiffness_float_repr = \
            convert_scalar_param_to_repr(marble_stiffness)
        node_stiffness_float_repr = convert_scalar_param_to_repr(
            node_stiffness)
        marble_attraction_float_repr = \
            convert_scalar_param_to_repr(marble_attraction)
        node_attraction_float_repr = \
            convert_scalar_param_to_repr(node_attraction)

        expected = f"MarbleEaterNode({repr(pos)},{repr(vel)},"\
            + f"{repr(acc)},{mass},NewtonianGravity(),"\
            + f"{marble_stiffness_float_repr}," \
            + f"{node_stiffness_float_repr},{marble_attraction_float_repr}," \
            + f"{node_attraction_float_repr},"\
            + f"{radius})"

        eater = MarbleEaterNode(pos, vel, acc, mass, attraction_function,
                                marble_stiffness, node_stiffness,
                                marble_attraction, node_attraction,
                                radius)

        result = repr(eater)
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
