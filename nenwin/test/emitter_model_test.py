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

Unit-tests for emitter_model.py.
"""

import unittest
import numpy as np

from nenwin.test.test_aux import ZERO, check_close
from nenwin.attraction_functions.attraction_functions import ConstantAttraction
from nenwin.auxliary import generate_stiffness_dict
from nenwin.particle import Particle
from nenwin.node import Marble
from nenwin.marble_emitter_node import MarbleEmitter, \
    Emitter, MarbleEmitterNode
from nenwin.emitter_model import ModelWithEmitters

ATTRACT_FUNCT = ConstantAttraction(0)


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
