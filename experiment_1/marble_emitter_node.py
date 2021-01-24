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
import torch
import abc

from experiment_1.marble_eater_node import MarbleEaterNode
from experiment_1.node import Marble, Node
from experiment_1.auxliary import distance
from experiment_1.constants import MAX_EMITTER_SPAWN_DIST


class MarbleEmitterNode(MarbleEaterNode):
    """
    A variant of MarbleEaterNodes that also can generate new Marbles.
    """

    def __init__(self,
                 pos: torch.Tensor,
                 vel: torch.Tensor,
                 acc: torch.Tensor,
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
        self.__raise_error_if_prototype_too_far_away(emitter.prototype)

    def __raise_error_if_prototype_too_far_away(self, prototype: Marble):
        distance_to_prototype = distance(self, prototype)
        if distance_to_prototype > self.radius + MAX_EMITTER_SPAWN_DIST:
            raise ValueError("prototype to emit further than "
                + "MAX_EMITTER_SPAWN_DIST from border of radius")

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
                 stored_mass: Optional[float] = 0,
                 initial_time_passed: Optional[float] = 0):
        """
        Arguments:
        * Prototype: instance of object to be copied when emitting a new
            instance.
        * delay: amount of time between consecutive emits are allowed.
        * stored_mass: initial amount of mass stored, that is depleted when
            emitting.
        * initial_time_passed: how much time since the last delay is
            initially counted, used to set time until first real emit.
        """
        self.__prototype = prototype.copy()
        self.__delay = delay
        self.__stored_mass = stored_mass
        self.__time_since_last_emit = initial_time_passed

    @property
    def prototype(self):
        return self.__prototype

    def can_emit(self) -> bool:
        can = (self.__time_since_last_emit >= self.__delay)

        output_mass = self.__prototype.mass
        if (output_mass >= 0):
            can = can and (self.__stored_mass >= output_mass)
        else:
            can = can and (self.__stored_mass <= output_mass)

        return can

    def emit(self) -> Node:
        if self.can_emit():
            self.__time_since_last_emit = 0
            self.__stored_mass -= self.__prototype.mass
            return self._create_particle()
        else:
            raise RuntimeError(
                "Cannot emit, delay not passed or too little mass stored")

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
                 delay: float,
                 stored_mass: Optional[float] = 0,
                 initial_time_passed: Optional[float] = 0):
        if not isinstance(prototype, Marble):
            raise ValueError("Prototype of MarbleEmitter must be a Marble")
        super().__init__(prototype, delay, stored_mass, initial_time_passed)

    def _create_particle(self) -> Marble:
        return self.prototype.copy()
