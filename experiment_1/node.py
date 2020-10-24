"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

The particle that is used to build the architecture of the network:
massfull 'nodes' that attract the marbles, and that way steer their path.
They themselves may or may not be attracted by the marbles as well.
"""
from __future__ import annotations
import numpy as np
from experiment_1.particle import PhysicalParticle
from experiment_1.marble import Marble

class Node(PhysicalParticle):
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
                 stiffness: float):
        """
        Initialize the node at the given position with the given acceleration,
        velocity and mass. The Node's attraction to any other Particle
        is described by the [attraction_function].
        The stiffness is a measure how moveable the Node is:
        at a stiffness of 0 if fully obeys Newtonian mechanics w.r.t. the
        forces that act on it, at a stiffness of 1 it is stationary independent
        of its environment. Any value inbetween scales the effect of forces
        on the acceleration of the Node proportionally.
        """
        super().__init__(pos, vel, acc, mass, attraction_function)
        if (stiffness < 0) or (stiffness > 1):
            raise ValueError("Stiffness must have a value in [0, 1]")
        self.__stiffness = stiffness

    @property
    def stiffness(self):
        return self.__stiffness

    def update_acceleration(self, forces: np.ndarray):
        """
        Decorates super's method to take stiffness into account.
        """
        if (self.__stiffness != 1):
            forces = (1 - self.__stiffness) * forces
            super().update_acceleration(forces)

class MarbleEaterNode(Node):
    """
    A node variant that consumes Marbles when they become within
    a certain radius of the Eater.
    Keeps track of total count of Marbles eater,
    and the data fields of the Marbles it ate.
    """

    def __init__(self,
                pos: np.ndarray,
                 vel: np.ndarray,
                 acc: np.ndarray,
                 mass: float,
                 attraction_function: callable,
                 stiffness: float,
                 radius: float):
        super().__init__(pos, vel, acc, mass, attraction_function, stiffness)
        self.__radius = radius
        self.__num_marbles_eaten: int = 0
        self.__marble_data_eaten = []

    @property
    def radius(self) -> float:
        return self.__radius

    @property
    def num_marbles_eaten(self) -> int:
        return self.__num_marbles_eaten

    @property
    def marble_data_eaten(self) -> List[object]:
        return self.__marble_data_eaten.copy()

    def eat(self, marble: Marble):
        self.__num_marbles_eaten += 1
        self.__marble_data_eaten.append(marble.datum)
        