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

Unit-tests for InitialValueParticle of particle.py.
"""
import unittest
import numpy as np
import torch

from experiment_1.particle import InitialValueParticle, PhysicalParticle
from experiment_1.attraction_functions.attraction_functions import Gratan
from test_aux import NUMERICAL_ABS_ACCURACY_REQUIRED
from test_aux import check_close
from test_aux import runge_kutta_4_step
from test_aux import check_named_parameters


class InitialValueParticleTestCase(unittest.TestCase):

    def setUp(self):
        pos = np.array([1, 3, 2])
        vel = np.array([5, 5, 1])
        acc = np.array([0, 0, 0])
        self.particle = InitialValueParticle(pos, vel, acc)

    def test_parameters(self):
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        particle = InitialValueParticle(pos, vel, acc)

        named_params = particle.named_parameters()
        expected_names = {'_InitialValueParticle__init_pos': pos,
                          '_InitialValueParticle__init_vel': vel,
                          '_InitialValueParticle__init_acc': acc}
        self.assertTrue(check_named_parameters(expected_names,
                                               tuple(named_params)))

    def test_gradients_pos(self):
        """
        Test if the gradients flow back from the pos to the initial_pos.
        """
        # Not a real loss function, just some random differentiable operations
        loss = torch.tensor([1, 1, 1], dtype=torch.float, requires_grad=True) 
        loss = loss / self.particle.pos
        loss = torch.sum(loss)
        
        loss.backward()
        self.assertIsNotNone(self.particle.init_pos.grad)


    def test_gradients_vel(self):
        """
        Test if the gradients flow back from the vel to the initial_vel.
        """
        # Not a real loss function, just some random differentiable operations
        vel = self.particle.vel
        loss = vel - torch.tensor([1, 1, 1], dtype=torch.float, requires_grad=True)
        loss = torch.sum(loss)
        loss.backward()
        self.assertIsNotNone(self.particle.init_vel.grad)

    def test_gradients_acc(self):
        """
        Test if the gradients flow back from the acc to the initial_acc.
        """
        # Not a real loss function, just some random differentiable operations
        loss = self.particle.acc - 1
        loss = torch.sum(loss)

        loss.backward()
        self.assertIsNotNone(self.particle.init_acc.grad)

    def test_reset(self):
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        particle = InitialValueParticle(pos, vel, acc)

        particle.pos = np.array([9, 9, 9])
        particle.vel = np.array([9, 9, 9])
        particle.acc = np.array([9, 9, 9])

        particle.reset()

        self.assertTrue(check_close(particle.pos.detach().numpy(), pos))
        self.assertTrue(check_close(particle.vel.detach().numpy(), vel))
        self.assertTrue(check_close(particle.acc.detach().numpy(), acc))

if __name__ == '__main__':
    unittest.main()
