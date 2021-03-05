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

Unit-tests for Marble of node.py.
"""
import unittest
import torch


from experiment_1.node import EmittedMarble
from experiment_1.auxliary import generate_stiffness_dict
from test_aux import ATTRACT_FUNCT


class MarbleTestCase(unittest.TestCase):

    def test_error_if_mass_not_tensor(self):
        pos = torch.tensor([1.])
        vel = torch.tensor([2.])
        acc = torch.tensor([3.])
        mass = 4
        datum = 5
        attraction_funct = ATTRACT_FUNCT
        stiffnesses = generate_stiffness_dict(0.6, 0.7, 0.8, 0.9)
        
        with self.assertRaises(ValueError):
            EmittedMarble(pos, vel, acc, mass, attraction_funct,
                          datum=datum, **stiffnesses)

    def test_error_if_mass_is_tensor(self):
        pos = torch.tensor([1.])
        vel = torch.tensor([2.])
        acc = torch.tensor([3.])
        mass = torch.tensor([4.])
        datum = 5
        attraction_funct = ATTRACT_FUNCT
        stiffnesses = generate_stiffness_dict(0.6, 0.7, 0.8, 0.9)
        
        assert mass.is_leaf, "Testcase broken!"

        with self.assertRaises(ValueError):
            EmittedMarble(pos, vel, acc, mass, attraction_funct,
                          datum=datum, **stiffnesses)

    def test_no_error_if_mass_is_non_leaf(self):
        pos = torch.tensor([1.])
        vel = torch.tensor([2.])
        acc = torch.tensor([3.])
        a = torch.tensor([4.], requires_grad=True)
        mass = 3*a
        datum = 5
        attraction_funct = ATTRACT_FUNCT
        stiffnesses = generate_stiffness_dict(0.6, 0.7, 0.8, 0.9)

        assert not mass.is_leaf, "Testcase broken"

        try:
            EmittedMarble(pos, vel, acc, mass, attraction_funct,
                          datum=datum, **stiffnesses)
        except ValueError as e:
            self.fail(
                f"Error raised by __init__, but input is valid. Error: {e}")

if __name__ == '__main__':
    unittest.main()
