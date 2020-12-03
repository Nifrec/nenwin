"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Unit-tests for marble_emitter_node.py.
"""

import unittest
import numpy as np

from test_aux import ZERO, ATTRACT_FUNCT, check_close
from experiment_1.auxliary import generate_stiffness_dict
from experiment_1.particle import Particle
from experiment_1.stiffness_particle import Marble
from experiment_1.marble_emitter_node import MarbleEmitter, \
    Emitter, MarbleEmitterNode

class MockEmitter(Emitter):
    def _create_particle(self):
        pass

class MockPrototype(Marble):

    def __init__(self):
        super().__init__(ZERO, ZERO, ZERO, 0, None, "TestDatum", 0, 0, 0, 0)
        self.copy_called = False

    def copy(self):
        self.copy_called = True
        return self
    

class MarbleEmitterTestCase(unittest.TestCase):
    def setUp(self):
        self.__mock_prototype = MockPrototype()

    def test_emit(self):
        emitter = MarbleEmitter(self.__mock_prototype, 0)
        emitter.emit()
        self.assertTrue(self.__mock_prototype.copy_called)

    def test_delay_1(self):
        """
        Corner case: can always emit at start.
        """
        emitter = MarbleEmitter(self.__mock_prototype, 10)
        self.assertTrue(emitter.can_emit())

    def test_delay_2(self):
        """
        Base case: cannot emit right after previous emit, when delay > 0.
        """
        emitter = MarbleEmitter(self.__mock_prototype, 10)
        emitter.emit()
        self.assertFalse(emitter.can_emit())

    def test_delay_3(self):
        """
        Base case: can emit second time after enough time passed.
        """
        delay = 10
        emitter = MarbleEmitter(self.__mock_prototype, delay)
        emitter.emit()
        emitter.register_time_passed(delay)
        self.assertTrue(emitter.can_emit())

    def test_delay_4(self):
        """
        Base case: can emit second time after *more* than enough time passed.
        """
        delay = 10
        emitter = MarbleEmitter(self.__mock_prototype, delay)
        emitter.emit()
        emitter.register_time_passed(delay + 10)
        self.assertTrue(emitter.can_emit())

    def test_emit_1(self):
        """
        Corner case: throw error when trying to emit 
        too early after previous emit.
        """
        delay = 10
        emitter = MarbleEmitter(self.__mock_prototype, delay)
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
        # self.assertTrue(check_close(expected.pos, result.pos))
        # self.assertTrue(check_close(expected.vel, result.vel))
        # self.assertTrue(check_close(expected.acc, result.acc))
        # self.assertEqual(expected.marble_stiffness, result.marble_stiffness)
        # self.assertEqual(expected.marble_attraction, result.marble_attraction)
        # self.assertEqual(expected.node_stiffness, result.node_stiffness)
        # self.assertEqual(expected.node_attraction, result.node_attraction)
        # self.assertIs(expected.datum, result.datum)


class MarbleEmitterNodeTestCase(unittest.TestCase):

    def test_copy(self):
        pos = np.array([1])
        vel = np.array([2])
        acc = np.array([3])
        mass = 4
        attraction_funct = ATTRACT_FUNCT
        stiffnesses = generate_stiffness_dict(0.5, 0.6, 0.7, 0.8)
        radius = 9
        emitter = MockEmitter(None, None)
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

        self.assertTrue(check_close(acc, copy.acc))
        self.assertTrue(check_close(vel, copy.vel))
        self.assertTrue(check_close(pos, copy.pos))
        self.assertEqual(mass, copy.mass)
        self.assertTrue(attraction_funct is copy._attraction_function)
        self.assertEqual(copy.marble_stiffness,
                         stiffnesses["marble_stiffness"])
        self.assertEqual(copy.node_stiffness, stiffnesses["node_stiffness"])
        self.assertEqual(copy.marble_attraction,
                         stiffnesses["marble_attraction"])
        self.assertEqual(copy.node_attraction, stiffnesses["node_attraction"])
        self.assertEqual(copy.radius, radius)
        self.assertEqual(copy.num_marbles_eaten, 0)
        self.assertIs(emitter, copy.emitter)

    

if __name__ == "__main__":
    unittest.main()
