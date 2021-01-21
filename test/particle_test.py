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

Unit-tests for Particle of particle.py.
"""
import unittest
import numpy as np
import torch

from experiment_1.particle import Particle
from experiment_1.particle import PhysicalParticle
from experiment_1.attraction_functions.attraction_functions import Gratan
from experiment_1.constants import DEVICE
from test_aux import NUMERICAL_ABS_ACCURACY_REQUIRED
from test_aux import check_close
from test_aux import runge_kutta_4_step
from test_aux import check_named_parameters


class ParticleTestCase(unittest.TestCase):

    def test_particle_getters(self):
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([0, 0, 0], dtype=torch.float)
        particle = Particle(pos, vel, acc)

        self.assertTrue(torch.allclose(particle.pos, pos),
                        "Pos getter")
        self.assertTrue(torch.allclose(particle.vel, vel),
                        "Vel getter")
        self.assertTrue(torch.allclose(particle.acc, acc),
                        "Acc getter")

    def test_particle_setters(self):
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([0, 0, 0], dtype=torch.float)
        particle = Particle(pos, vel, acc)

        pos2 = torch.tensor([1, 2, 3], dtype=torch.float)
        vel2 = torch.tensor([4, 5, 6], dtype=torch.float)
        acc2 = torch.tensor([7, 8, 9], dtype=torch.float)

        particle.pos = pos2
        particle.vel = vel2
        particle.acc = acc2

        self.assertTrue(torch.allclose(particle.pos, pos2),
                        "Pos setter")
        self.assertTrue(torch.allclose(particle.vel, vel2),
                        "Vel setter")
        self.assertTrue(torch.allclose(particle.acc, acc2),
                        "Acc setter")

    def test_particle_setters_dim_check(self):
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([0, 0, 0], dtype=torch.float)
        particle = Particle(pos, vel, acc)

        pos2 = torch.tensor([1, 2, 3, 2], dtype=torch.float)
        vel2 = torch.tensor([], dtype=torch.float)
        acc2 = torch.tensor([7, 8], dtype=torch.float)

        def set_pos(pos):
            particle.pos = pos

        def set_vel(vel):
            particle.vel = vel

        def set_acc(acc):
            particle.acc = acc

        self.assertRaises(RuntimeError, set_pos, pos2)
        self.assertRaises(RuntimeError, set_vel, vel2)
        self.assertRaises(RuntimeError, set_acc, acc2)

    def test_particle_init_dim_check(self):
        ok1 = torch.tensor([1, 3, 2], dtype=torch.float)
        nok = torch.tensor([1, 1, 1, 1], dtype=torch.float)
        ok2 = torch.tensor([0, 0, 0], dtype=torch.float)

        self.assertRaises(ValueError, Particle, ok1, nok, ok2)
        self.assertRaises(ValueError, Particle, nok, ok1, ok2)
        self.assertRaises(ValueError, Particle, ok2, ok1, nok)

    def test_movement_1(self):
        """
        Base case: positive acceleration.
        """
        pos = torch.tensor([10, 20], dtype=torch.float)
        vel = torch.tensor([0, 0], dtype=torch.float)
        acc = torch.tensor([1, 1], dtype=torch.float)
        p = Particle(pos, vel, acc)
        for _ in range(100):
            p.update_movement(0.01)

        expected = runge_kutta_4_step(pos, vel, acc)
        
        self.assertTrue(check_close(p.pos, expected[0]))
        self.assertTrue(check_close(p.vel, expected[1]))
        # Should not have changed
        self.assertTrue(check_close(p.acc, acc))  

    def test_movement_2(self):
        """
        Base case: deacceleration.
        """
        pos = torch.tensor([-10, -100], dtype=torch.float)
        vel = torch.tensor([100, 1000], dtype=torch.float)
        acc = torch.tensor([0, -91], dtype=torch.float)
        p = Particle(pos, vel, acc)
        for _ in range(100):
            p.update_movement(0.01)

        expected = runge_kutta_4_step(pos, vel, acc)
        self.assertTrue(check_close(expected[1], p.vel))
        self.assertTrue(check_close(expected[0], p.pos))
        # Should not have changed
        self.assertTrue(check_close(acc, p.acc))  

    def test_movement_3(self):
        """
        Corner case: no vel and no acc => no movement.
        """
        pos = torch.tensor([-10, -100], dtype=torch.float)
        vel = torch.tensor([0, 0], dtype=torch.float)
        acc = torch.tensor([0, 0], dtype=torch.float)
        p = Particle(pos, vel, acc)
        for _ in range(100):
            p.update_movement(0.01)

        expected = runge_kutta_4_step(pos, vel, acc)
        self.assertTrue(check_close(acc, p.acc))
        self.assertTrue(check_close(vel, p.vel))
        self.assertTrue(check_close(pos, p.pos))

    def test_copy(self):
        pos = torch.tensor([-10, -100], dtype=torch.float)
        vel = torch.tensor([0, 0], dtype=torch.float)
        acc = torch.tensor([0, 0], dtype=torch.float)
        original = Particle(pos, vel, acc)
        copy = original.copy()

        self.assertFalse(copy is original)

        self.assertTrue(check_close(acc, copy.acc))
        self.assertTrue(check_close(vel, copy.vel))
        self.assertTrue(check_close(pos, copy.pos))

if __name__ == '__main__':
    unittest.main()
