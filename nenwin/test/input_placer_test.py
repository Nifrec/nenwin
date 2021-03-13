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

Unit tests for inputplacer
"""
import abc
import unittest
import numpy as np
from typing import Iterable, Optional, Sequence, List
import math
import torch

from nenwin.node import Marble
from nenwin.attraction_functions.attraction_functions import NewtonianGravity

class InputPlacerTestCase(unittest.TestCase):
    
    def test_placer_maker(self):
        input_pos = [0]
        input_dimensions = [1]
        placer = PhiInputPlacer(input_pos, input_dimensions)
        self.assertAlmostEqual(placer.input_pos, input_pos)
        self.assertAlmostEqual(placer.input_region_sizes, input_dimensions)
        
if __name__ == '__main__':
    unittest.main()