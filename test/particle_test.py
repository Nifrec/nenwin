"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Unit-tests for particle.py.
"""
import unittest
import numpy as np

from experiment_1.particle import Particle
from experiment_1.particle import PhysicalParticle
from experiment_1.attraction_functions.attraction_functions import Gratan
from test_aux import NUMERICAL_ABS_ACCURACY_REQUIRED
from test_aux import check_close
from test_aux import runge_kutta_4_step

class ParticleTestCase(unittest.TestCase):

    def test_particle_getters(self):
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        particle = Particle(pos, vel, acc)

        self.assertTrue(np.allclose(particle.pos, pos), "Pos getter")
        self.assertTrue(np.allclose(particle.vel, vel), "Vel getter")
        self.assertTrue(np.allclose(particle.acc, acc), "Acc getter")

    def test_particle_setters(self):
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        particle = Particle(pos, vel, acc)

        pos2 = np.array([1, 2, 3])
        vel2 = np.array([4, 5, 6])
        acc2 = np.array([7, 8, 9])

        particle.pos = pos2
        particle.vel = vel2
        particle.acc = acc2

        self.assertTrue(np.allclose(particle.pos, pos2), "Pos setter")
        self.assertTrue(np.allclose(particle.vel, vel2), "Vel setter")
        self.assertTrue(np.allclose(particle.acc, acc2), "Acc setter")

    def test_particle_setters_dim_check(self):
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        particle = Particle(pos, vel, acc)

        pos2 = np.array([1, 2, 3, 2])
        vel2 = np.array([])
        acc2 = np.array([7, 8])

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
        ok1 = np.array([1, 3, 2])
        nok = np.array([1, 1, 1, 1])
        ok2 = np.array([0, 0, 0])

        self.assertRaises(ValueError, Particle, ok1, nok, ok2)
        self.assertRaises(ValueError, Particle, nok, ok1, ok2)
        self.assertRaises(ValueError, Particle, ok2, ok1, nok)

    def test_movement_1(self):
        """
        Base case: positive acceleration.
        """
        pos = np.array([10, 20])
        vel = np.array([0, 0])
        acc = np.array([1, 1])
        p = Particle(pos, vel, acc)
        for _ in range(100):
            p.update_movement(0.01)

        expected = runge_kutta_4_step(pos, vel, acc)
        self.assertTrue(check_close(p.acc, acc))  # Should not have changed
        self.assertTrue(check_close(p.pos, expected[0]))
        self.assertTrue(check_close(p.vel, expected[1]))

    def test_movement_2(self):
        """
        Base case: deacceleration.
        """
        pos = np.array([-10, -100])
        vel = np.array([100, 1000])
        acc = np.array([0, -91])
        p = Particle(pos, vel, acc)
        for _ in range(100):
            p.update_movement(0.01)

        expected = runge_kutta_4_step(pos, vel, acc)
        self.assertTrue(check_close(acc, p.acc))  # Should not have changed
        self.assertTrue(check_close(expected[1], p.vel))
        self.assertTrue(check_close(expected[0], p.pos))

    def test_movement_3(self):
        """
        Corner case: no vel and no acc => no movement.
        """
        pos = np.array([-10, -100])
        vel = np.array([0, 0])
        acc = np.array([0, 0])
        p = Particle(pos, vel, acc)
        for _ in range(100):
            p.update_movement(0.01)

        expected = runge_kutta_4_step(pos, vel, acc)
        self.assertTrue(check_close(acc, p.acc))
        self.assertTrue(check_close(vel, p.vel))
        self.assertTrue(check_close(pos, p.pos))


class PhysicalParticleTestCase(unittest.TestCase):

    def test_mass_setter_getter(self):
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        mass = 0
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        self.assertEqual(particle.mass, mass)

        mass2 = 10
        particle.mass = mass2

        self.assertEqual(particle.mass, mass2)

    def test_update_acceleration_1(self):
        """
        Base case: apply two forces in different directions.
        """
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([10, 9, 0])
        mass = 2
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        forces = np.array([[1, 2, 3], [-4, -5, -7]])
        expected_net_force = np.array([-3, -3, -4])
        expected_acc = expected_net_force / mass

        particle.update_acceleration(forces)

        self.assertTrue(check_close(particle.acc, expected_acc))

    def test_update_acceleration_2(self):
        """
        Corner case: empty force set (that is allowed!)
        """
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([10, 9, 0])
        mass = 2
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        forces = np.array([])
        expected_acc = np.zeros(3)

        particle.update_acceleration(forces)

        self.assertTrue(check_close(particle.acc, expected_acc))

    def test_update_acceleration_force_vector_dim_error(self):
        """
        Input [forces] does not have 2 dimensions.
        """
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        mass = 2
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        forces = np.array([1, 2, 3])
        self.assertRaises(ValueError,
                          particle.update_acceleration,
                          forces)

    def test_update_acceleration_force_vector_dim_error_1(self):
        """
        Input [forces] has 2 dimensions but wrong length force vector 
        (too long).
        """
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        mass = 2
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        forces = np.array([[1, 2, 3, 4]])
        self.assertRaises(ValueError,
                          particle.update_acceleration,
                          forces)

    def test_update_acceleration_force_vector_dim_error_2(self):
        """
        Input [forces] has 2 dimensions but wrong length force vector 
        (too short).
        """
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        mass = 2
        particle = PhysicalParticle(pos, vel, acc, mass, lambda: None)

        forces = np.array([[1, 2]])
        self.assertRaises(ValueError,
                          particle.update_acceleration,
                          forces)

    def test_compute_attraction_force_to_1(self):
        """
        Positive attraction direction.
        """
        attraction_funct = Gratan()
        pos1 = np.array([1, 1, 1])
        vel1 = np.array([0, 0, 0])
        acc1 = np.array([0, 0, 0])
        mass1 = 2
        p1 = PhysicalParticle(pos1, vel1, acc1, mass1, attraction_funct)

        pos2 = np.array([0, 0, 0])
        vel2 = np.array([0, 0, 0])
        acc2 = np.array([0, 0, 0])
        mass2 = 3
        p2 = PhysicalParticle(pos2, vel2, acc2, mass2, attraction_funct)

        # p2 is pulled towards positive all directions
        expected = np.array([np.sqrt(3)/3, np.sqrt(3)/3, np.sqrt(3)/3]) \
            * attraction_funct.compute_attraction(p1, p2)
        result = p1.compute_attraction_force_to(p2)
        self.assertTrue(check_close(result, expected))


    def test_compute_attraction_force_to_2(self):
        """
        Negative attraction direction.
        """
        attraction_funct = Gratan()
        pos1 = np.array([1, 0, 0])
        vel1 = np.array([0, 0, 0])
        acc1 = np.array([0, 0, 0])
        mass1 = 2
        p1 = PhysicalParticle(pos1, vel1, acc1, mass1, attraction_funct)

        pos2 = np.array([10, 0, 0])
        vel2 = np.array([0, 0, 0])
        acc2 = np.array([0, 0, 0])
        mass2 = 3
        p2 = PhysicalParticle(pos2, vel2, acc2, mass2, attraction_funct)

        # -1 is for the direction, p2 is pulled towards negative x-direction
        expected = np.array([-1, 0, 0]) \
            * attraction_funct.compute_attraction(p1, p2)
        result = p1.compute_attraction_force_to(p2)
        self.assertTrue(check_close(result, expected))


if __name__ == '__main__':
    unittest.main()
