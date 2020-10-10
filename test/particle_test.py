"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée

Unit-tests for particle.py.
"""
import unittest
import numpy as np
from src.particle import Particle
from src.particle import PhysicalParticle

class ConcreteTestParicle(Particle):
    """
    Clone of particle, but not abstract.
    Hence just a dummy to test.
    """
    def update(self, time_passed: float):
        """Override abstract method"""
        pass

class ParticleTestCase(unittest.TestCase):

    def test_particle_getters(self):
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        particle = ConcreteTestParicle(pos, vel, acc)

        self.assertTrue(np.allclose(particle.pos, pos), "Pos getter")
        self.assertTrue(np.allclose(particle.vel, vel), "Vel getter")
        self.assertTrue(np.allclose(particle.acc, acc), "Acc getter")

    def test_particle_setters(self):
        pos = np.array([1, 3, 2])
        vel = np.array([1, 1, 1])
        acc = np.array([0, 0, 0])
        particle = ConcreteTestParicle(pos, vel, acc)

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
        particle = ConcreteTestParicle(pos, vel, acc)

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

        self.assertRaises(ValueError, ConcreteTestParicle, ok1, nok, ok2)
        self.assertRaises(ValueError, ConcreteTestParicle, nok, ok1, ok2)
        self.assertRaises(ValueError, ConcreteTestParicle, ok2, ok1, nok)

class PhysicalParticleTestCase(unittest.TestCase):
    def test_gravity_1(self):
        assert False, "TODO"

    def test_update_1(self):
        assert False, "TODO"

if __name__ == '__main__':
    unittest.main()
