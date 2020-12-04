"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

A variant of the Node that can create new Marbles, after having absorbed
sufficient other Marbles.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import abc

from experiment_1.marble_eater_node import MarbleEaterNode
from experiment_1.node import Marble, Node


class MarbleEmitterNode(MarbleEaterNode):
    """
    A variant of MarbleEaterNodes that also can generate new Marbles.
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
                 node_attraction,
                 radius: float,
                 emitter: Emitter):
        super().__init__(pos, vel, acc, mass,
                         attraction_function,
                         marble_stiffness,
                         node_stiffness,
                         marble_attraction,
                         node_attraction,
                         radius)
        self.__emitter = emitter

    def eat(self, marble: Marble):
        self.__emitter.eat_mass(marble.mass)
        super().eat(marble)

    @property
    def emitter(self):
        return self.__emitter

    def copy(self) -> MarbleEmitterNode:
        """
        Create copy of this MarbleEmitterNode,
        but reset the number of marbles eaten of the copy to 0.
        The copy will use the same emitter instance.
        """
        return MarbleEmitterNode(self.pos,
                               self.vel,
                               self.acc,
                               self.mass,
                               self._attraction_function,
                               self.marble_stiffness,
                               self.node_stiffness,
                               self.marble_attraction,
                               self.node_attraction,
                               self.radius,
                               self.emitter)


class Emitter(abc.ABC):

    def __init__(self,
                 prototype: Node,
                 delay: float,
                 stored_mass: Optional[float] = 0):
        self.__stored_mass = stored_mass
        self.__delay = delay
        self.__time_since_last_emit = float("inf")

    def can_emit(self) -> bool:
        return (self.__time_since_last_emit >= self.__delay)


    def emit(self) -> Node:
        if self.__time_since_last_emit >= self.__delay:
            self.__time_since_last_emit = 0
            return self._create_particle()
        else:
            raise RuntimeError("Cannot emit, delay not completely passed")

    @abc.abstractmethod
    def _create_particle(self) -> Node:
        pass

    def eat_mass(self, mass: float):
        self.__stored_mass += mass

    def register_time_passed(self, time_passed: float):
        self.__time_since_last_emit += time_passed
 

class MarbleEmitter(Emitter):

    def __init__(self,
                 prototype: Marble,
                 delay: flat,
                 stored_mass: Optional[float] = 0):
        if not isinstance(prototype, Marble):
            raise ValueError("Prototype of MarbleEmitter must be a Marble")
        super().__init__(prototype, delay, stored_mass)
        self.__prototype = prototype.copy()

    def _create_particle(self) -> Marble:
        return self.__prototype.copy()
