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

--------------------------------------------------------------------------------

Integration test for using PyTorch's autograd on a NenwinModel.
Mostly intended to resolve any errors.
"""

import unittest
import torch
import torch.optim as optim

from experiment_1.model import NenwinModel
from experiment_1.node import Marble
from experiment_1.marble_eater_node import MarbleEaterNode
from test_aux import ATTRACT_FUNCT


class BackPropTestCase(unittest.TestCase):


    def setUp(self):
        pos = torch.tensor([0], dtype=torch.float)
        vel = pos.clone().detach()
        acc = pos.clone().detach()
        self.marble = Marble(pos, vel, acc, mass=1,
                        attraction_function=ATTRACT_FUNCT,
                        datum=None)

    def test_backprop_single_marble(self):
        """
        Base case: run backward() on a single Marble, and check if any errors
        occur, and if the gradient is set.
        """
        self.marble.update_acceleration(torch.tensor([[.5]]))
        self.marble.update_movement(time_passed=1)
        loss = torch.tensor([1]) - self.marble.pos
        loss.backward()

        not_expected = torch.tensor([0], dtype=torch.float)
        self.assertFalse(torch.isclose(not_expected,
                                       self.marble.init_pos.grad))

    # def test_backprop_clone(self):
    #     """
    #     Base case: run backward() on a clone of a Marble, 
    #     and check if also the gradient of the original is set.
    #     """
    #     marble = self.marble
    #     marble.update_acceleration(torch.tensor([[.5]]))
    #     marble.update_movement(time_passed=1)
    #     loss = torch.tensor([1]) - marble._Particle__pos
    #     loss.backward()

    #     not_expected = torch.tensor([0], dtype=torch.float)
    #     self.assertFalse(torch.isclose(not_expected,
    #                                    marble._Particle__pos.grad))    

if __name__ == "__main__":
    unittest.main()
