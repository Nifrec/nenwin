"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

The particle that is used to represent individual data entries.
"""
from __future__ import annotations
import numpy as np
from experiment_1.particle import PhysicalParticle

class Marble(PhysicalParticle):
    """
    The particle that is used to build the architecture of the network:
    massfull 'nodes' that attract the marbles, and that way steer their path.
    They themselves may or may not be attracted by the marbles as well.
    """

    def __init__(self,
                 pos: np.ndarray,
                 vel: np.ndarray,
                 acc: np.ndarray,
                 mass: float,
                 attraction_function: callable,
                 datum: object):
        """
        Initialize the Marble at the given position with the given acceleration,
        velocity and mass. The Marble's attraction to any other Particle
        is described by the [attraction_function].
        
        The datum is the object that the Marble represent. 
        What form it may have is not specified.
        """
        super().__init__(pos, vel, acc, mass, attraction_function)
        self.__datum = datum

    @property
    def datum(self):
        return self.__datum
