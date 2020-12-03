"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

A variant of the Node that can 'consume' Marbles if they come close.
"""
from __future__ import annotations
import numpy as np
from experiment_1.stiffness_particle import Marble, Node


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
                 marble_stiffness: float,
                 node_stiffness: float,
                 marble_attraction: float,
                 node_attraction: float,
                 radius: float):
        super().__init__(pos, vel, acc, mass,
                         attraction_function,
                         marble_stiffness,
                         node_stiffness,
                         marble_attraction,
                         node_attraction)
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

    def copy(self) -> MarbleEaterNode:
        """
        Create copy of this MarbleEaterNode,
        but reset the number of marbles eaten of the copy to 0.
        """
        return MarbleEaterNode(self.pos,
                               self.vel,
                               self.acc,
                               self.mass,
                               self._attraction_function,
                               self.marble_stiffness,
                               self.node_stiffness,
                               self.marble_attraction,
                               self.node_attraction,
                               self.radius)
