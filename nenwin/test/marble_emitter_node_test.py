"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
March 2021

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

Unit-tests for marble_emitter_node.py.
"""
import unittest
import numpy as np
from typing import Optional, Tuple
import torch

from nenwin.test.test_aux import ZERO, ATTRACT_FUNCT, convert_scalar_param_to_repr
from nenwin.test.test_aux import check_named_parameters
from nenwin.attraction_functions.attraction_functions \
    import NewtonianGravity
from nenwin.auxliary import generate_stiffness_dict
from nenwin.node import Marble, Node
from nenwin.marble_emitter_node import MarbleEmitter, \
    Emitter, MarbleEmitterNode, MarbleEmitterVariablePosition
from nenwin.constants import MAX_EMITTER_SPAWN_DIST
from nenwin.test.test_aux import check_close


class MockEmitter(Emitter):

    def __init__(self, prototype: Node, delay: float,
                 stored_mass: Optional[float]=0,
                 initial_time_passed: Optional[float] = 0):
        super().__init__(prototype, delay, stored_mass=stored_mass,
                         initial_time_passed=initial_time_passed)
        self.was_reset = False

    def _create_particle(self):
        pass

    def reset(self):
        self.was_reset = True


class MockPrototype(Marble):

    def __init__(self, mass: Optional[float] = 0):
        super().__init__(ZERO, ZERO, ZERO, mass, None, "TestDatum", 0, 0, 0, 0)
        self.copy_called = False

    def copy(self):
        self.copy_called = True
        return super().copy()


class MarbleEmitterNodeTestCase(unittest.TestCase):

    def test_spawn_distance(self):
        """
        Should raise error when the distance of the spawned Marble
        is greater than specified in constants.py
        """
        node_pos = ZERO
        radius = 1
        marble_pos = node_pos + radius + 2*MAX_EMITTER_SPAWN_DIST
        other_settings = {"vel": ZERO, "acc": ZERO, "mass": 0,
                          "attraction_function": ATTRACT_FUNCT}
        other_settings.update(generate_stiffness_dict(0, 0, 0, 0))
        prototype_marble = Marble(pos=marble_pos, datum=None, **other_settings)
        emitter = MockEmitter(prototype_marble, 0)

        def gen_node(): return MarbleEmitterNode(pos=node_pos, emitter=emitter,
                                                 radius=radius, **other_settings)

        self.assertRaises(ValueError, gen_node)

    def test_spawn_location_1(self):
        """
        Base case: moving Node. 
        Also the prototype should follow the movement.
        This should be done automatically when calling MarbleEmitterNode.emit().
        Using a MarbleEmitterVariablePosition.
        """
        vel = torch.tensor([1.0])
        radius = 10
        eps = 10e-5
        prototype = Marble(ZERO, ZERO, ZERO, 0,
                           None, None, 0, 0, 0, 0)
        emitter = MarbleEmitterVariablePosition(
            prototype, 0, 10, rel_prototype_pos=torch.tensor([radius + eps]))
        emitter_node = MarbleEmitterNode(ZERO, vel, ZERO, 0, None,
                                         0, 0, 0, 0,
                                         radius=radius, emitter=emitter)
        emitter_node.update_movement(1)

        self.assertTrue(check_close(emitter_node.pos, torch.tensor([1.0])))
        expected_marble_pos = torch.tensor([1.0]) + radius + eps
        self.assertTrue(check_close(
            emitter_node.emit().pos, expected_marble_pos))

    def test_spawn_location_2(self):
        """
        Base case: stationary Node.
        Using a normal MarbleEmitter.
        """
        vel = torch.tensor([1.0])
        radius = 10
        eps = 10e-5
        prototype = Marble(ZERO + radius + eps, ZERO, ZERO, 0,
                           None, None, 0, 0, 0, 0)
        emitter = MarbleEmitter(prototype, 0, 10)
        emitter_node = MarbleEmitterNode(ZERO, vel, ZERO, 0, None,
                                         0, 0, 0, 0,
                                         radius=radius, emitter=emitter)

        self.assertTrue(check_close(emitter_node.pos, ZERO))
        self.assertTrue(check_close(emitter.emit().pos, ZERO + radius + eps))

    def test_copy(self):
        """
        Test copying a MarbleEmitterNode.
        Trainable init_ variables should keep a reference to the original.
        The emitter should be copied, and *not* be a reference to the original.
        This is because the stored mass of the new Node 
        may differ over time from the original!
        """

        pos = torch.tensor([1], dtype=torch.float)
        vel = torch.tensor([2], dtype=torch.float)
        acc = torch.tensor([3], dtype=torch.float)
        mass = 4
        attraction_funct = ATTRACT_FUNCT
        stiffnesses = generate_stiffness_dict(0.5, 0.6, 0.7, 0.8)
        radius = 9
        emitter = MockEmitter(MockPrototype(), 0)
        original = MarbleEmitterNode(pos,
                                     vel,
                                     acc,
                                     mass,
                                     attraction_funct,
                                     radius=radius,
                                     emitter=emitter,
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
        self.assertEqual(copy.radius, radius)
        self.assertEqual(copy.num_marbles_eaten, 0)
        self.assertIsNot(emitter, copy.emitter)

    def test_copy_same_reference(self):
        """
        Implementation test: the copy should have exactly the same
        learnable parameters as the original. 
        This ensures backpropagation will propagate from the copies to the
        original.
        """
        original = create_particle()
        copy = original.copy()

        self.assertIsNot(copy, original)
        self.assertIs(copy.init_pos, original.init_pos)
        self.assertIs(copy.init_vel, original.init_vel)
        self.assertIs(copy.init_acc, original.init_acc)
        self.assertIs(copy.marble_stiffness, original.marble_stiffness)
        self.assertIs(copy.node_stiffness, original.node_stiffness)
        self.assertIs(copy.marble_attraction, original.marble_attraction)
        self.assertIs(copy.node_attraction, original.node_attraction)

    def test_copy_type(self):
        """
        The copy should be an instance of a MarbleEmitterNode.
        """
        original = create_particle()
        copy = original.copy()

        self.assertIsInstance(copy, MarbleEmitterNode)

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

        emitter = MockEmitter(MockPrototype(), 0)

        expected = f"MarbleEmitterNode({repr(pos)},{repr(vel)},"\
            + f"{repr(acc)},{mass},NewtonianGravity(),"\
            + f"{marble_stiffness_float_repr}," \
            + f"{node_stiffness_float_repr},{marble_attraction_float_repr}," \
            + f"{node_attraction_float_repr},"\
            + f"{radius}," \
            + f"{repr(emitter)})"

        eater = MarbleEmitterNode(pos, vel, acc, mass, attraction_function,
                                  marble_stiffness, node_stiffness,
                                  marble_attraction, node_attraction,
                                  radius, emitter)

        result = repr(eater)
        self.assertEqual(expected, result)

    def test_reset(self):
        emitter_node = create_particle()
        emitter_node.reset()
        self.assertTrue(emitter_node._MarbleEmitterNode__emitter.was_reset)


def create_particle() -> MarbleEmitterNode:
    pos = torch.tensor([1], dtype=torch.float)
    vel = torch.tensor([2], dtype=torch.float)
    acc = torch.tensor([3], dtype=torch.float)
    mass = 4
    attraction_funct = ATTRACT_FUNCT
    stiffnesses = generate_stiffness_dict(0.5, 0.6, 0.7, 0.8)
    radius = 9
    emitter = MockEmitter(MockPrototype(), 0)
    output = MarbleEmitterNode(pos,
                               vel,
                               acc,
                               mass,
                               attraction_funct,
                               radius=radius,
                               emitter=emitter,
                               **stiffnesses)
    return output


if __name__ == "__main__":
    unittest.main()
