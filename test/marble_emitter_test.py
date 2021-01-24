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

Unit-tests for marble_emitter_node.py.
"""

import unittest
import numpy as np
from typing import Optional
import torch

from test_aux import ZERO, ATTRACT_FUNCT, convert_scalar_param_to_repr
from experiment_1.attraction_functions.attraction_functions \
    import NewtonianGravity
from experiment_1.auxliary import generate_stiffness_dict
from experiment_1.particle import Particle
from experiment_1.node import Marble
from experiment_1.marble_emitter_node import MarbleEmitter, \
    Emitter, MarbleEmitterNode
from experiment_1.constants import MAX_EMITTER_SPAWN_DIST


class MockEmitter(Emitter):
    def _create_particle(self):
        pass


class MockPrototype(Marble):

    def __init__(self, mass: Optional[float] = 0):
        super().__init__(ZERO, ZERO, ZERO, mass, None, "TestDatum", 0, 0, 0, 0)
        self.copy_called = False

    def copy(self):
        self.copy_called = True
        return self


class MarbleEmitterTestCase(unittest.TestCase):
    def setUp(self):
        self.__mock_prototype = MockPrototype()
        self.__mass = 1
        self.__massfull_prototype = MockPrototype(self.__mass)

    def test_emit(self):
        emitter = MarbleEmitter(self.__mock_prototype, 0)
        emitter.emit()
        self.assertTrue(self.__mock_prototype.copy_called)

    def test_delay_1(self):
        """
        Base case: can always emit at start if initial_time_passed high enough.
        """
        emitter = MarbleEmitter(self.__mock_prototype, 10,
                                initial_time_passed=10)
        self.assertTrue(emitter.can_emit())

    def test_delay_2(self):
        """
        Base case: cannot emit right after previous emit, when delay > 0.
        """
        delay = 10
        emitter = MarbleEmitter(self.__mock_prototype, delay)
        emitter.register_time_passed(delay)
        emitter.emit()
        self.assertFalse(emitter.can_emit())

    def test_delay_3(self):
        """
        Base case: can emit second time after enough time passed.
        """
        delay = 10
        emitter = MarbleEmitter(self.__mock_prototype, delay)
        emitter.register_time_passed(delay)
        emitter.emit()
        emitter.register_time_passed(delay)
        self.assertTrue(emitter.can_emit())

    def test_delay_4(self):
        """
        Base case: can emit second time after *more* than enough time passed.
        """
        delay = 10
        emitter = MarbleEmitter(self.__mock_prototype, delay)
        emitter.register_time_passed(delay + 10)
        self.assertTrue(emitter.can_emit())

    def test_delay_5(self):
        """
        Base case: cannot emit at start if initial_time_passed is too low.
        """
        emitter = MarbleEmitter(self.__mock_prototype, 10,
                                initial_time_passed=0)
        self.assertFalse(emitter.can_emit())

    def test_can_emit_massfull_1(self):
        """
        Base case: cannot emit massfull Marble when no mass stored.
        """
        emitter = MarbleEmitter(self.__massfull_prototype, 0, stored_mass=0)
        self.assertFalse(emitter.can_emit())

    def test_can_emit_massfull_2(self):
        """
        Base case: can emit after adding sufficient mass.
        """
        emitter = MarbleEmitter(self.__massfull_prototype, 0, stored_mass=0)
        emitter.eat_mass(self.__mass)
        self.assertTrue(emitter.can_emit())

    def test_can_emit_massfull_3(self):
        """
        Base case: can emit negative mass when negative mass stored.
        """
        neg_mass_prototype = MockPrototype(-1)
        emitter = MarbleEmitter(neg_mass_prototype, 0, stored_mass=-1)
        self.assertTrue(emitter.can_emit())

    def test_can_emit_massfull_4(self):
        """
        Base case: can emit if initial mass is sufficient.
        """
        emitter = MarbleEmitter(self.__massfull_prototype,
                                0,
                                stored_mass=self.__mass)
        self.assertTrue(emitter.can_emit())

    def test_can_emit_massfull_5(self):
        """
        Corner case: cannot emit negative mass if own mass is positive.
        """
        emitter = MarbleEmitter(MockPrototype(-1),
                                0,
                                stored_mass=1)
        self.assertFalse(emitter.can_emit())

    def test_emit_1(self):
        """
        Corner case: throw error when trying to emit 
        too early after previous emit.
        """
        delay = 10
        emitter = MarbleEmitter(self.__mock_prototype, delay)
        emitter.register_time_passed(delay)
        emitter.emit()
        emitter.register_time_passed(delay/2)
        self.assertRaises(RuntimeError, emitter.emit)

    def test_emit_2(self):
        """
        Base case: check if emitted Marble is similar to prototype.
        """
        emitter = MarbleEmitter(self.__mock_prototype, 0)
        result = emitter.emit()

        expected = self.__mock_prototype
        self.assertIs(expected, result)

    def test_emit_mass_stored_1(self):
        """
        Base case: emitting positive mass decreases mass stored.
        """
        emitter = MarbleEmitter(self.__massfull_prototype,
                                0,
                                stored_mass=self.__mass)
        emitter.emit()
        self.assertEqual(emitter._Emitter__stored_mass, 0)

    def test_emit_mass_stored_2(self):
        """
        Base case: emitting negative mass increases mass stored.
        """
        emitter = MarbleEmitter(MockPrototype(-1),
                                0,
                                stored_mass=-1)
        emitter.emit()
        self.assertEqual(emitter._Emitter__stored_mass, 0)

    def test_emit_mass_stored_3(self):
        """
        Corner case: emitting 0 mass does not affect mass stored.
        """
        emitter = MarbleEmitter(self.__mock_prototype,
                                0,
                                stored_mass=13)
        emitter.emit()
        self.assertEqual(emitter._Emitter__stored_mass, 13)

    def test_emit_massfull_1(self):
        """
        Base case: cannot emit massfull Marble when no mass stored.
        """
        emitter = MarbleEmitter(self.__massfull_prototype, 0, stored_mass=0)
        self.assertRaises(RuntimeError, emitter.emit)

    def test_emit_massfull_2(self):
        """
        Base case: can emit massfull Marble when sufficient mass stored.
        """
        emitter = MarbleEmitter(self.__massfull_prototype,
                                0,
                                stored_mass=self.__mass)
        result = emitter.emit()

        expected = self.__massfull_prototype
        self.assertIs(expected, result)

    def test_emit_massfull_3(self):
        """
        Base case: can emit negaitve massfull Marble when negative mass stored.
        (Should not raise any errors)
        """
        neg_mass_prototype = MockPrototype(-1)
        emitter = MarbleEmitter(neg_mass_prototype, 0, stored_mass=-1)

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
        prototype = Marble(pos, vel, acc, mass,
                           attraction_function, None,
                           marble_stiffness, node_stiffness,
                           marble_attraction, node_attraction)
        delay = 8.1
        stored_mass = 9.1
        initial_time_passed = 11.1

        expected = f"MarbleEmitter({repr(prototype)},{delay},"\
            +f"{stored_mass},{initial_time_passed})"

        emitter = MarbleEmitter(prototype, delay,
                                stored_mass, initial_time_passed)
        result = repr(emitter)

        self.assertEqual(expected, result)
        


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

    def test_copy(self):
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
        self.assertAlmostEqual(copy.node_stiffness, stiffnesses["node_stiffness"])
        self.assertAlmostEqual(copy.marble_attraction,
                         stiffnesses["marble_attraction"])
        self.assertAlmostEqual(copy.node_attraction, stiffnesses["node_attraction"])
        self.assertEqual(copy.radius, radius)
        self.assertEqual(copy.num_marbles_eaten, 0)
        self.assertIs(emitter, copy.emitter)

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


if __name__ == "__main__":
    unittest.main()
