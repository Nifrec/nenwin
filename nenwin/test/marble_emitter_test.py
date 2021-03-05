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

Unit-tests for Emitter, MarbleEmitter and MarbleEmitterVariablePosition
from marble_emitter_node.py.
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
from nenwin.node import EmittedMarble, Marble
from nenwin.marble_emitter_node import MarbleEmitter, \
    Emitter, MarbleEmitterNode, MarbleEmitterVariablePosition
from nenwin.constants import MAX_EMITTER_SPAWN_DIST
from nenwin.test.test_aux import check_close
from marble_emitter_node_test import MockPrototype




class MarbleEmitterTestCase(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.__mock_prototype = MockPrototype()
        self.__mass = 1
        self.__massfull_prototype = MockPrototype(self.__mass)

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

    def test_emit_output_type(self):
        """
        Output should be a EmittedMarble, not just a normal Marble.
        """
        prototype = Marble(ZERO, ZERO, ZERO, 0, None, None)
        delay = 0
        emitter = MarbleEmitter(prototype, delay)
        output = emitter.emit()
        self.assertIsInstance(output, EmittedMarble)

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

    def test_emit_mass_stored_1(self):
        """
        Base case: emitting positive mass decreases mass stored.
        """
        emitter = MarbleEmitter(self.__massfull_prototype,
                                0,
                                stored_mass=self.__mass)
        emitter.emit()
        self.assertEqual(emitter._Emitter__stored_mass.item(), 0)

    def test_emit_mass_stored_2(self):
        """
        Base case: emitting negative mass increases mass stored.
        """
        emitter = MarbleEmitter(MockPrototype(-1),
                                0,
                                stored_mass=-1)
        emitter.emit()
        self.assertEqual(emitter._Emitter__stored_mass.item(), 0)

    def test_emit_mass_stored_3(self):
        """
        Corner case: emitting 0 mass does not affect mass stored.
        """
        emitter = MarbleEmitter(self.__mock_prototype,
                                0,
                                stored_mass=13)
        emitter.emit()
        self.assertEqual(emitter._Emitter__stored_mass.item(), 13)

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
        try:
            emitter.emit()
        except:
            self.fail("Operation should not raise errors")

    def test_emit_massfull_3(self):
        """
        Base case: can emit negative massfull Marble when negative mass stored.
        (Should not raise any errors)
        """
        neg_mass_prototype = MockPrototype(-1)
        emitter = MarbleEmitter(neg_mass_prototype, 0, stored_mass=-1)
        try:
            emitter.emit()
        except:
            self.fail("Operation should not raise errors")

    def test_repr_1(self):
        """
        Base case: no mass absorbed or time passed.
        """
        emitter, prototype, delay, stored_mass, initial_time_passed = \
            self.setup_full_emitter()

        expected = f"MarbleEmitter({repr(prototype)}," \
            + f"{convert_scalar_param_to_repr(delay)},"\
            + f"{convert_scalar_param_to_repr(stored_mass)},"\
            + f"{convert_scalar_param_to_repr(initial_time_passed)})"

        emitter = MarbleEmitter(prototype, delay,
                                stored_mass, initial_time_passed)
        result = repr(emitter)

        self.assertEqual(expected, result)

    def test_repr_2(self):
        """
        Base case: if mass absorbed and/or time passed,
        do still return the initial values (these can be trained).
        """
        emitter, prototype, delay, stored_mass, initial_time_passed = \
            self.setup_full_emitter()

        expected = f"MarbleEmitter({repr(prototype)}," \
            + f"{convert_scalar_param_to_repr(delay)},"\
            + f"{convert_scalar_param_to_repr(stored_mass)},"\
            + f"{convert_scalar_param_to_repr(initial_time_passed)})"

        # These should not affect the output
        emitter.eat_mass(10)
        emitter.register_time_passed(10)

        result = repr(emitter)

        self.assertEqual(expected, result)

    def test_named_params(self):
        emitter, prototype, delay, stored_mass, initial_time_passed = \
            self.setup_full_emitter()

        named_params = emitter.named_parameters()
        expected_names = {
            '_Emitter__delay': delay,
            '_Emitter__init_stored_mass': stored_mass,
            '_Emitter__inital_time_passed': initial_time_passed}
        self.assertTrue(check_named_parameters(expected_names,
                                               tuple(named_params)))

    def test_grads_mass(self):
        """
        A Marble [eaten_marble] is eaten by the MarbleEmitter.
        A second Marble m2 is emitted by the MarbleEmitter.
        The Marble m2 undergoes some movement and attraction,
        and backpropagating thereafter should 
        create a gradient in eaten_marble.mass.
        """
        emitter, prototype, delay, stored_mass, initial_time_passed = \
            self.setup_full_emitter()

        eaten_marble = Marble(ZERO, ZERO, ZERO, 10, None, None, 0, 0, 0, 0)

        emitter.eat_mass(eaten_marble.mass)
        emitter.register_time_passed(20)
        output_marble = emitter.emit()
        output_marble.update_movement(30)

        meh = torch.tensor([1.0], dtype=torch.float, requires_grad=True) \
            + output_marble.mass
        loss = torch.sum(meh)
        loss.backward(create_graph=True)
        print(output_marble.mass)

        print(output_marble.mass.grad)
        self.assertIsNotNone(emitter._Emitter__init_stored_mass.grad)
        self.assertIsNotNone(eaten_marble.mass.grad)

    def setup_full_emitter(self) -> Tuple[Emitter, Marble, float, float, float]:
        """
        Create an instance of a MarbleEmitter with a full prototype Marble
        (not a mock Marble). Also returns the delay, stored_mass
        and initial_time_passed, 
        which have arbitrary nonzero floating-point values.

        Returns:
        * A MarbleEmitter instance
        * A Marble, the prototype associated with the emitter
        * The delay of the emitter
        * The initally stored mass of the emitter
        * The initial time passed of the emitter
        """
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

        emitter = MarbleEmitter(prototype, delay,
                                stored_mass, initial_time_passed)

        return emitter, prototype, delay, stored_mass, initial_time_passed

    def test_copy_values_1(self):
        """
        Check if the copy has the same values as the original.
        Base case: original still has init values.
        """
        original, prototype, delay, stored_mass, initial_time_passed = \
            self.setup_full_emitter()
        copy = original.copy()
        self.assertAlmostEqual(copy.init_stored_mass, stored_mass)
        self.assertAlmostEqual(copy.init_time_passed, initial_time_passed)
        self.assertAlmostEqual(copy.delay, delay)
        self.assertIs(copy.prototype.init_pos, prototype.init_pos)

    def test_copy_values_2(self):
        """
        Check if the copy has the same values as the original.
        Base case: original's working values changed,
        copy should still have initial values.
        """
        original, prototype, delay, stored_mass, initial_time_passed = \
            self.setup_full_emitter()
        original.eat_mass(100)
        original.register_time_passed(100)
        copy = original.copy()
        self.assertAlmostEqual(copy.init_stored_mass, stored_mass)
        self.assertAlmostEqual(copy.init_time_passed, initial_time_passed)
        self.assertAlmostEqual(copy.delay, delay)
        self.assertIs(copy.prototype, prototype)

    def test_copy_same_reference(self):
        """
        Implementation test: the copy should have exactly the same
        learnable parameters as the original. 
        This ensures backpropagation will propagate from the copies to the
        original.
        """
        original, _, _, _, _ = self.setup_full_emitter()
        copy = original.copy()

        self.assertIsNot(copy, original)
        self.assertIs(copy.init_stored_mass, original.init_stored_mass)
        self.assertIs(copy.delay, original.delay)
        self.assertIs(copy.init_time_passed, original.init_time_passed)
        self.assertIs(copy.prototype, original.prototype)

    def test_copy_type(self):
        """
        The copy should be an instance of a MarbleEmitter.
        """
        original, _, _, _, _ = self.setup_full_emitter()
        copy = original.copy()

        self.assertIsInstance(copy, MarbleEmitter)


class MarbleEmitterVariablePositionTestCase(unittest.TestCase):

    def test_emit_with_location(self):
        """
        MarbleEmitterNodes typically do not emit Marbles at their own position,
        but at the boundary of their radius. 
        Since MarbleEmitterNodes can move, also the position of the emitted
        Marble may vary with time, hence MarbleEmitters should be able
        to set the position of the emitted Marble.
        """
        mass = 10
        prototype = Marble(ZERO, ZERO, ZERO, mass, None, None, 0, 0, 0, 0)
        emitter = MarbleEmitterVariablePosition(prototype,
                                                0,
                                                stored_mass=mass)
        spawn_pos = torch.tensor([1], dtype=torch.float)
        result = emitter.emit(spawn_pos)

        self.assertTrue(torch.allclose(spawn_pos, result.pos),
                        f"{spawn_pos} != {result.pos}")

    def test_emit_with_rel_pos(self):
        """
        Emitted Marbles can have a fixed initial pos set.
        """
        mass = 10
        prototype = Marble(ZERO, ZERO, ZERO, mass, None, None, 0, 0, 0, 0)
        rel_prototype_pos = torch.tensor([11.3])
        emitter = MarbleEmitterVariablePosition(
            prototype, 0, stored_mass=mass, rel_prototype_pos=rel_prototype_pos)

        spawn_pos = torch.tensor([1], dtype=torch.float)
        result = emitter.emit(spawn_pos)

        self.assertTrue(torch.allclose(
            spawn_pos + rel_prototype_pos, result.pos),
            f"{spawn_pos} != {result.pos}")

    def test_repr(self):
        (prototype, delay, stored_mass, initial_time_passed,
                  relative_prototype_pos, emitter) = self.set_up_full_emitter()

        expected = f"MarbleEmitterVariablePosition({repr(prototype)}," \
            + f"{convert_scalar_param_to_repr(delay)},"\
            + f"{convert_scalar_param_to_repr(stored_mass)},"\
            + f"{convert_scalar_param_to_repr(initial_time_passed)},"\
            + f"{relative_prototype_pos})"

        result = repr(emitter)

        self.assertEqual(expected, result)

    def test_copy(self):
        """
        Check if the copy has the same values as the original.
        Base case: original still has init values.
        """
        (prototype, delay, stored_mass, initial_time_passed,
                  rel_prototype_pos, emitter) = self.set_up_full_emitter()
        copy = emitter.copy()
        self.assertIsInstance(copy, MarbleEmitterVariablePosition)
        self.assertAlmostEqual(copy.init_stored_mass, stored_mass)
        self.assertAlmostEqual(copy.init_time_passed, initial_time_passed)
        self.assertAlmostEqual(copy.delay, delay)
        self.assertTrue(check_close(copy.rel_prototype_pos, rel_prototype_pos))
        self.assertIs(copy.prototype.init_pos, prototype.init_pos)

    def set_up_full_emitter(self) -> Tuple:
        prototype = Marble(ZERO, ZERO, ZERO, 0, None, None, 0, 0, 0, 0)
        delay = 1.1
        stored_mass = 1.2
        initial_time_passed = 1.3
        relative_prototype_pos = torch.tensor([1.4])
        emitter = MarbleEmitterVariablePosition(
            prototype, delay, stored_mass, initial_time_passed,
            relative_prototype_pos)

        output = (prototype, delay, stored_mass, initial_time_passed,
                  relative_prototype_pos, emitter)
        return output

if __name__ == "__main__":
    unittest.main()
