"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Unit-tests for emitter_model.py.
"""

import unittest
import numpy as np

from test_aux import ZERO, TestAttractionFunction, check_close
from experiment_1.auxliary import generate_stiffness_dict
from experiment_1.particle import Particle
from experiment_1.node import Marble
from experiment_1.marble_emitter_node import MarbleEmitter, \
    Emitter, MarbleEmitterNode
from experiment_1.emitter_model import ModelWithEmitters

ATTRACT_FUNCT = TestAttractionFunction(0)


class EmitterModelTestCase(unittest.TestCase):

    def setUp(self):
        self.__prototype = Marble(
            ZERO, ZERO, ZERO, 1, ATTRACT_FUNCT, None, 0, 0, 0, 0)

    def gen_emitter_node_with_delay(self, delay: float):
        emitter = MarbleEmitter(self.__prototype, delay)
        emitter_node = MarbleEmitterNode(np.array([1]), ZERO, ZERO,
                                         1, ATTRACT_FUNCT,
                                         0, 0, 0, 0, 0.1, emitter)
        return emitter, emitter_node

    def test_time_updated(self):
        """
        Test if during a timestep, the passed time is also passed on to
        the Emitters.
        """
        delay = 10
        emitter, emitter_node = self.gen_emitter_node_with_delay(delay)
        model = ModelWithEmitters([emitter_node])
        timestep = 5
        model.make_timestep(timestep)

        self.assertEqual(timestep, emitter._Emitter__time_since_last_emit)

    def test_marble_added_1(self):
        """
        Base case: when an Emitter can emit, at any point during a timestep,
        it should add an Marble.
        """
        delay = 10
        emitter, emitter_node = self.gen_emitter_node_with_delay(delay)
        model = ModelWithEmitters([emitter_node])
        timestep = 5
        model.make_timestep(timestep)
        model.make_timestep(timestep)

        self.assertEqual(len(model.marbles), 1)

        model.make_timestep(timestep)

        self.assertEqual(len(model.marbles), 1)

    def test_marble_added_2(self):
        """
        Base case: when an Emitter can emit, it should add an Marble,
        also a second time.
        """
        delay = 10
        emitter, emitter_node = self.gen_emitter_node_with_delay(delay)
        model = ModelWithEmitters([emitter_node])

        timestep = 11
        for step in (1, 2):
            model.make_timestep(timestep)
            self.assertEqual(len(model.marbles), step)

    def test_marble_added_3(self):
        """
        Corner case: should not emit before delay satisfied.
        """
        delay = 10
        emitter, emitter_node = self.gen_emitter_node_with_delay(delay)
        model = ModelWithEmitters([emitter_node])

        timestep = 9
        model.make_timestep(timestep)

        self.assertEqual(len(model.marbles), 0)


if __name__ == "__main__":
    unittest.main()
