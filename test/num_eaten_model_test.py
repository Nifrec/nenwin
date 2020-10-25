"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

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

Author: Lulof Pirée
October 2020

Unit-tests for class NumMarblesEatenAsOutputModel of model.py.
"""
import unittest
import numpy as np
from typing import Tuple

from experiment_1.node import Node, MarbleEaterNode
from experiment_1.marble import Marble
from experiment_1.model import NumMarblesEatenAsOutputModel
from experiment_1.attraction_functions.attraction_functions \
    import NewtonianGravity
from experiment_1.particle import PhysicalParticle
from test_aux import check_close


class NumMarblesEatenAsOutputModelTestCase(unittest.TestCase):
    def setUp(self):
        self.attract_funct = NewtonianGravity()

    def test_output_1(self):
        """
        Base case: single marble created within radius of eater node,
        should become output after 1 step.
        """
        zero = np.array([0])
        eater = MarbleEaterNode(zero, zero, zero, 10,
                                self.attract_funct, 1, 10)
        marble = Marble(np.array([9]), zero, zero, 1, self.attract_funct, None)
        model = NumMarblesEatenAsOutputModel([eater], 1, [marble])

        model.run(1)

        result = model._produce_outputs()
        expected = np.array([1])
        self.assertTrue(check_close(result, expected))

    def test_output_2(self):
        """
        Corner case: no eaternodes, empty array as output.
        """
        zero = np.array([0])
        marble = Marble(np.array([9]), zero, zero, 1, self.attract_funct, None)
        model = NumMarblesEatenAsOutputModel([], 1, [marble])

        model.run(1)

        result = model._produce_outputs()
        expected = np.array([])
        self.assertTrue(check_close(result, expected))

    def test_output_3(self):
        """
        Base case: single marble created outside radius of eater node,
        and not cominig close. Output should remain 0.
        """
        zero = np.array([0])
        eater = MarbleEaterNode(zero, zero, zero, 10,
                                self.attract_funct, 1, 10)
        # Moving too fast to be pulled back.
        marble = Marble(np.array([11]), np.array([100]),
                        zero, 1, self.attract_funct, None)
        model = NumMarblesEatenAsOutputModel([eater], 1, [marble])

        model.run(1)

        result = model._produce_outputs()
        expected = np.array([0])
        self.assertTrue(check_close(result, expected))

    def test_output_4(self):
        """
        Base case: node close but not within eater, becomes pulled in.
        """
        zero = np.array([0])
        eater = MarbleEaterNode(zero, zero, zero, 100,
                                self.attract_funct, 1, 10)
        marble = Marble(np.array([11]), zero,
                        zero, 1, self.attract_funct, None)
        model = NumMarblesEatenAsOutputModel([eater], 1, [marble])

        model.run(1)

        result = model._produce_outputs()
        expected = np.array([0])
        self.assertTrue(check_close(result, expected))

        model.run(10)

        result = model._produce_outputs()
        expected = np.array([1])
        self.assertTrue(check_close(result, expected))



if __name__ == '__main__':
    unittest.main()
