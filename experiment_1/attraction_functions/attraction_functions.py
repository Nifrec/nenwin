"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Collection of functions to be used to compute the attraction force
between particles.
"""
import abc
import numpy as np
from experiment_1.particle import PhysicalParticle
import warnings

class AttractionFunction(abc.ABC):
    """
    Abstract class providing interface for all AttractionFunctions.
    This branch of classes are simply functions with a standardized interface.
    Given two particles, subclasses of this class are supposed to return
    a numeric value for the attraction of the first particle onto the second
    (Note that in many cases this might be symmetric, but this is not required).
    """

    def __call__(self,
                 first_particle: PhysicalParticle,
                 second_particle: PhysicalParticle
                 ) -> float:
        return self.compute_attraction(first_particle, second_particle)

    @abc.abstractmethod
    def compute_attraction(self,
                 first_particle: PhysicalParticle,
                 second_particle: PhysicalParticle
                 ) -> float:
        pass

class Gratan(AttractionFunction):
    """
    Gravity-like function based on tanh:
        p1, p2 -> p1.mass * p2.mass * (1 - |tanh(radius)|)
    """

    def compute_attraction(self,
                 first_particle: PhysicalParticle,
                 second_particle: PhysicalParticle
                 ) -> float:
        radius = np.linalg.norm(first_particle.pos - second_particle.pos)
        return first_particle.mass * second_particle.mass *(1-abs(np.tanh(radius)))
        
class NewtonianGravity(AttractionFunction):
    """
    Newton's non-relativistic gravity formula:
        p1, p2 -> p1.mass * p2.mass / radius
    
    Note that radius should not equal 0 (division by 0).
    """

    def compute_attraction(self,
                 first_particle: PhysicalParticle,
                 second_particle: PhysicalParticle
                 ) -> float:
        warnings.warn("NewtonianGravity not yet unit-tested!")
        radius = np.linalg.norm(first_particle.pos - second_particle.pos)
        return first_particle.mass * second_particle / radius