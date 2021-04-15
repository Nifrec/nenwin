"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Teun Schilperoort
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
from nenwin.input_placer import PhiInputPlacer

class InputPlacerTestCase(unittest.TestCase):
    
    def test_placer_maker(self):
        input_pos = [0]
        input_dimensions = [1]
        placer = PhiInputPlacer(input_pos, input_dimensions)
        self.assertAlmostEqual(placer.input_pos, input_pos)
        self.assertAlmostEqual(placer.input_region_sizes, input_dimensions)
        
    def test_marblelize_data_output(self):
        input_pos = [0,0]
        input_dimensions = [2,2]
        wanted_output = "(Marble(tensor([1.0195, 0.2794]),tensor([0., 0.]),tensor([0., 0.]),1.0,NewtonianGravity(),[1, 2],1.0,0.0,0.0,1.0), Marble(tensor([1.5098, 1.1397]),tensor([0., 0.]),tensor([0., 0.]),1.0,NewtonianGravity(),[2, 3],1.0,0.0,0.0,1.0), Marble(tensor([0.5293, 1.4190]),tensor([0., 0.]),tensor([0., 0.]),1.0,NewtonianGravity(),[4, 5],1.0,0.0,0.0,1.0))"
        placer = PhiInputPlacer(input_pos, input_dimensions)
        tensors = placer.marblelize_data([[1,2], [2,3], [4,5]])
        self.assertMultiLineEqual(str(tensors), wanted_output) 

    def test_marblelize_data_output_2(self):
        input_pos = [1, 3, 6, 9, 1]
        input_dimensions = [1,4,7,3,2]
        wanted_output = "(Marble(tensor([1.7625, 5.2131, 8.5820, 9.6190, 1.1262]),tensor([0., 0., 0., 0., 0.]),tensor([0., 0., 0., 0., 0.]),1.0,NewtonianGravity(),[0, 2, 0, 0, 0],1.0,0.0,0.0,1.0), Marble(tensor([ 1.8813,  6.1066, 10.7910, 10.8095,  2.0631]),tensor([0., 0., 0., 0., 0.]),tensor([0., 0., 0., 0., 0.]),1.0,NewtonianGravity(),[1, 2, 3, 2, 3],1.0,0.0,0.0,1.0), Marble(tensor([ 1.6438,  4.3197,  6.3730, 11.4285,  2.1893]),tensor([0., 0., 0., 0., 0.]),tensor([0., 0., 0., 0., 0.]),1.0,NewtonianGravity(),[4, 5, 8, 9, 1],1.0,0.0,0.0,1.0))"
        placer = PhiInputPlacer(input_pos, input_dimensions)
        tensors = placer.marblelize_data(input_data = [[0,2,0,0,0], [1,2,3,2,3], [4,5,8,9,1]], vel = torch.tensor([0,0,0,0,0], dtype=torch.float), acc = torch.tensor([0,0,0,0,0], dtype=torch.float))
        self.assertMultiLineEqual(str(tensors), wanted_output)
                               
if __name__ == '__main__':
    unittest.main()