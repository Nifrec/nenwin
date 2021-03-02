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
from typing import Optional, Tuple
import torch

from test_aux import ZERO, ATTRACT_FUNCT, convert_scalar_param_to_repr
from test_aux import check_named_parameters
from experiment_1.attraction_functions.attraction_functions \
    import NewtonianGravity
from experiment_1.auxliary import generate_stiffness_dict
from experiment_1.node import Marble
from experiment_1.marble_emitter_node import MarbleEmitter, \
    Emitter, MarbleEmitterNode, MarbleEmitterVariablePosition
from experiment_1.constants import MAX_EMITTER_SPAWN_DIST
from test_aux import check_close


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


@unittest.skip("First debug Abstract Emitter")
class MarbleEmitterTestCase(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
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
        result = emitter.emit()

        expected = neg_mass_prototype
        self.assertIs(expected, result)

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

    def test_grads(self):
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
        emitter._Emitter__stored_mass
        output_marble._PhysicalParticle__mass

        meh = torch.tensor([1, 1], dtype=torch.float, requires_grad=True) \
            * output_marble.mass
        torch.sum(meh).backward(create_graph=True)
        print(output_marble.mass)

        print(output_marble.mass.grad)
        self.assertIsNotNone(output_marble.mass.grad)
        # self.assertIsNotNone(emitter._Emitter__init_stored_mass.grad)
        # # self.assertIsNotNone(emitter._Emitter__delay.grad)
        # # self.assertIsNotNone(emitter._Emitter__init_stored_mass.grad)
        # # self.assertIsNotNone(emitter._Emitter__inital_time_passed.grad)
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
        self.assertIs(copy.prototype, prototype)

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

    def test_repr(self):
        prototype = Marble(ZERO, ZERO, ZERO, 0, None, None, 0, 0, 0, 0)
        delay = 1.1
        stored_mass = 1.2
        initial_time_passed = 1.3
        relative_prototype_pos = torch.tensor([1.4])

        expected = f"MarbleEmitterVariablePosition({repr(prototype)}," \
            + f"{convert_scalar_param_to_repr(delay)},"\
            + f"{convert_scalar_param_to_repr(stored_mass)},"\
            + f"{convert_scalar_param_to_repr(initial_time_passed)},"\
            + f"{relative_prototype_pos})"

        emitter = MarbleEmitterVariablePosition(prototype, delay,
                                stored_mass, initial_time_passed, relative_prototype_pos)
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

    def test_spawn_location_1(self):
        """
        Base case: moving Node. Also the prototype should follow the movement.
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
        emitter_node.update_movement(1)

        self.assertTrue(check_close(emitter_node.pos, torch.tensor([1.0])))
        expected_marble_pos = torch.tensor([1.0]) + radius + eps
        self.assertTrue(check_close(emitter.emit().pos, expected_marble_pos))

    def test_spawn_location_2(self):
        """
        Base case: stationary Node.
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
