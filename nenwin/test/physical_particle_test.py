"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Copyright (C) 2020 Lulof Pirée, 

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

Unit-tests for PhysicalParticle of particle.py.
"""
import unittest
import numpy as np
import torch

from nenwin.particle import Particle
from nenwin.particle import PhysicalParticle
from nenwin.attraction_functions.attraction_functions \
    import NewtonianGravity
from nenwin.attraction_functions.attraction_functions import Gratan
from nenwin.constants import DEVICE
from nenwin.test.test_aux import check_close


class PhysicalParticleTestCase(unittest.TestCase):

    def test_mass_setter_getter(self):
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([0, 0, 0], dtype=torch.float)
        mass = 0
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        self.assertEqual(particle.mass, mass)

        mass2 = torch.nn.Parameter(torch.tensor([10.0]), requires_grad=True)
        particle.set_mass(mass2)

        self.assertEqual(particle.mass, mass2)

    def test_update_acceleration_1(self):
        """
        Base case: apply two forces in different directions.
        """
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([10, 9, 0], dtype=torch.float)
        mass = 2
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        forces = torch.tensor([[1, 2, 3], [-4, -5, -7]], dtype=torch.float)
        expected_net_force = torch.tensor([-3, -3, -4], dtype=torch.float)
        expected_acc = expected_net_force / mass

        particle.update_acceleration(forces)

        self.assertTrue(torch.allclose(particle.acc, expected_acc))

    def test_update_acceleration_2(self):
        """
        Corner case: empty force set (that is allowed!)
        """
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([10, 9, 0], dtype=torch.float)
        mass = 2
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        forces = torch.tensor([], dtype=torch.float)
        expected_acc = torch.zeros(3)

        particle.update_acceleration(forces)

        self.assertTrue(torch.allclose(particle.acc, expected_acc))

    def test_update_acceleration_3(self):
        """
        Corner case: negative mass, should not affect direction.
        """
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([10, 9, 0], dtype=torch.float)
        mass = -2
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        forces = torch.tensor([[1, 1, 1]], dtype=torch.float)
        expected_acc = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float)

        particle.update_acceleration(forces)

        self.assertTrue(torch.allclose(particle.acc, expected_acc))

    def test_update_acceleration_force_vector_dim_error(self):
        """
        Input [forces] does not have 2 dimensions.
        """
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([0, 0, 0], dtype=torch.float)
        mass = 2
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        forces = torch.tensor([1, 2, 3], dtype=torch.float)
        self.assertRaises(ValueError,
                          particle.update_acceleration,
                          forces)

    def test_update_acceleration_force_vector_dim_error_1(self):
        """
        Input [forces] has 2 dimensions but wrong length force vector 
        (too long).
        """
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([0, 0, 0], dtype=torch.float)
        mass = 2
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        forces = torch.tensor([[1, 2, 3, 4]], dtype=torch.float)
        self.assertRaises(ValueError,
                          particle.update_acceleration,
                          forces)

    def test_update_acceleration_force_vector_dim_error_2(self):
        """
        Input [forces] has 2 dimensions but wrong length force vector 
        (too short).
        """
        pos = torch.tensor([1, 3, 2], dtype=torch.float)
        vel = torch.tensor([1, 1, 1], dtype=torch.float)
        acc = torch.tensor([0, 0, 0], dtype=torch.float)
        mass = 2
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        forces = torch.tensor([[1, 2]], dtype=torch.float)
        self.assertRaises(ValueError,
                          particle.update_acceleration,
                          forces)

    def test_compute_attraction_force_to_1(self):
        """
        Positive attraction direction.
        """
        attraction_funct = Gratan()
        pos1 = torch.tensor([1, 1, 1], dtype=torch.float)
        vel1 = torch.tensor([0, 0, 0], dtype=torch.float)
        acc1 = torch.tensor([0, 0, 0], dtype=torch.float)
        mass1 = 2
        p1 = PhysicalParticle(pos1, vel1, acc1, mass1, attraction_funct)

        pos2 = torch.tensor([0, 0, 0], dtype=torch.float)
        vel2 = torch.tensor([0, 0, 0], dtype=torch.float)
        acc2 = torch.tensor([0, 0, 0], dtype=torch.float)
        mass2 = 3
        p2 = PhysicalParticle(pos2, vel2, acc2, mass2, attraction_funct)

        # p2 is pulled towards positive all directions
        expected = torch.tensor([np.sqrt(3)/3, np.sqrt(3)/3, np.sqrt(3)/3], dtype=torch.float) \
            * attraction_funct.compute_attraction(p1, p2)
        result = p1.compute_attraction_force_to(p2)
        self.assertTrue(torch.allclose(result, expected))

    def test_compute_attraction_force_to_2(self):
        """
        Negative attraction direction.
        """
        attraction_funct = Gratan()
        pos1 = torch.tensor([1, 0, 0], dtype=torch.float)
        vel1 = torch.tensor([0, 0, 0], dtype=torch.float)
        acc1 = torch.tensor([0, 0, 0], dtype=torch.float)
        mass1 = 2
        p1 = PhysicalParticle(pos1, vel1, acc1, mass1, attraction_funct)

        pos2 = torch.tensor([10, 0, 0], dtype=torch.float)
        vel2 = torch.tensor([0, 0, 0], dtype=torch.float)
        acc2 = torch.tensor([0, 0, 0], dtype=torch.float)
        mass2 = 3
        p2 = PhysicalParticle(pos2, vel2, acc2, mass2, attraction_funct)

        # -1 is for the direction, p2 is pulled towards negative x-direction
        expected = torch.tensor([-1, 0, 0], dtype=torch.float) \
            * attraction_funct.compute_attraction(p1, p2)
        result = p1.compute_attraction_force_to(p2)
        self.assertTrue(torch.allclose(result, expected))

    def test_compute_attraction_force_to_3(self):
        """
        One particle with negative mass, one with positive mass.
        """
        attraction_funct = Gratan()
        pos1 = torch.tensor([1, 0, 0], dtype=torch.float)
        vel1 = torch.tensor([0, 0, 0], dtype=torch.float)
        acc1 = torch.tensor([0, 0, 0], dtype=torch.float)
        mass1 = 2
        p1 = PhysicalParticle(pos1, vel1, acc1, mass1, attraction_funct)

        pos2 = torch.tensor([10, 0, 0], dtype=torch.float)
        vel2 = torch.tensor([0, 0, 0], dtype=torch.float)
        acc2 = torch.tensor([0, 0, 0], dtype=torch.float)
        mass2 = -3
        p2 = PhysicalParticle(pos2, vel2, acc2, mass2, attraction_funct)

        # p2 is repelled from p1,
        # so the force should be in the positive x-direction.
        expected = torch.tensor([1, 0, 0], dtype=torch.float) \
            * attraction_funct.compute_attraction(p1, p2)

        result = p1.compute_attraction_force_to(p2)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_copy(self):
        pos = torch.tensor([1], dtype=torch.float)
        vel = torch.tensor([2], dtype=torch.float)
        acc = torch.tensor([3], dtype=torch.float)
        mass = 4
        attraction_funct = Gratan()
        original = PhysicalParticle(pos, vel, acc, mass, attraction_funct)
        copy = original.copy()

        self.assertFalse(copy is original)

        self.assertTrue(torch.allclose(acc, copy.acc))
        self.assertTrue(torch.allclose(vel, copy.vel))
        self.assertTrue(torch.allclose(pos, copy.pos))
        self.assertEqual(mass, copy.mass)
        self.assertTrue(attraction_funct is copy._attraction_function)

    def test_copy_with_mass_grad(self):
        """
        Gradients of mass should be copied as well.
        """
        pos = torch.tensor([1], dtype=torch.float)
        vel = torch.tensor([2], dtype=torch.float)
        acc = torch.tensor([3], dtype=torch.float)
        mass = 4
        attraction_funct = Gratan()
        some_particle = PhysicalParticle(pos, vel, acc, mass, attraction_funct)

        some_tensor = torch.tensor([4.0], requires_grad=True)
        loss = torch.sum(some_particle.mass + some_tensor)
        loss.backward()
        self.assertIsNotNone(some_particle.mass.grad)
        self.assertIsNotNone(some_tensor.grad)

        another_particle = some_particle.copy()
        self.assertIs(some_particle.mass.grad, another_particle.mass.grad)

    def test_parameters(self):
        pos = torch.tensor([1], dtype=torch.float)
        vel = torch.tensor([2], dtype=torch.float)
        acc = torch.tensor([3], dtype=torch.float)
        mass = 4
        attraction_funct = Gratan()
        particle = PhysicalParticle(pos, vel, acc, mass, attraction_funct)

        named_params = particle.named_parameters()
        expected_names = {'_PhysicalParticle__mass': mass}
        for name, param in named_params:
            if name in set(expected_names.keys()):
                expected_value = expected_names.pop(name)
                self.assertTrue(torch.allclose(torch.tensor(
                    expected_value, dtype=torch.float), param))
        self.assertEqual(len(expected_names), 0)

    def test_repr(self):
        pos = torch.tensor([0], dtype=torch.float)
        vel = torch.tensor([1], dtype=torch.float)
        acc = torch.tensor([2], dtype=torch.float)
        mass = 3.0
        attraction_funct = NewtonianGravity()

        expected = f"PhysicalParticle({repr(pos)},{repr(vel)},"\
            + f"{repr(acc)},{mass},NewtonianGravity(),{repr(DEVICE)})"
        result = repr(PhysicalParticle(pos, vel, acc, mass, attraction_funct))
        self.assertEqual(expected, result)

    def test_mass_gradients_1(self):
        """
        The mass of a PhysicalParticle should collect gradients
        when computing any arbitrary loss whose computation uses
        the mass.
        """
        pos = torch.tensor([10, 10], dtype=torch.float)
        vel = torch.tensor([0, 0], dtype=torch.float)
        acc = torch.tensor([0, 0], dtype=torch.float)
        mass = 10
        particle = PhysicalParticle(pos, vel, acc, mass, None)

        v = torch.tensor([1.0, 1.0], requires_grad=True)
        loss = torch.sum(v + particle.mass)
        loss.backward()
        self.assertIsNotNone(v.grad)
        self.assertIsNotNone(particle._PhysicalParticle__mass.grad)


if __name__ == '__main__':
    unittest.main()
