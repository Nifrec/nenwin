"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
March 2021

A variant of the Node that can create new Marbles, after having absorbed
sufficient other Marbles.
"""
from __future__ import annotations
from typing import Optional
import torch
import abc
import re
import torch.nn as nn

from nenwin.marble_eater_node import MarbleEaterNode
from nenwin.node import Marble, Node, EmittedMarble
from nenwin.auxliary import distance
from nenwin.constants import MAX_EMITTER_SPAWN_DIST
from nenwin.particle import create_param


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

    def __repr__(self) -> str:
        output = super().__repr__()
        output = re.sub("Eater", "Emitter", output)
        output = re.sub("\)$", f",{repr(self.__emitter)})", output)
        return output

    def eat(self, marble: Marble):
        self.__emitter.eat_mass(marble.mass)
        super().eat(marble)

    def emit(self) -> Node:
        if self.__emitter.can_emit():
            if isinstance(self.emitter, MarbleEmitterVariablePosition):
                return self.__emitter.emit(self.pos)
            else:
                return self.__emitter.emit()
        else:
            raise RuntimeError("Unable to emit a Marble")

    @property
    def emitter(self):
        return self.__emitter.copy()

    def copy(self) -> MarbleEmitterNode:
        """
        Create copy of this MarbleEmitterNode,
        but reset the number of marbles eaten of the copy to 0.
        The copy will use a different emitter instance.
        """
        output = MarbleEmitterNode(self.init_pos,
                                   self.init_vel,
                                   self.init_acc,
                                   self.mass,
                                   self._attraction_function,
                                   self.marble_stiffness,
                                   self.node_stiffness,
                                   self.marble_attraction,
                                   self.node_attraction,
                                   self.radius,
                                   self.emitter.copy())
        output.adopt_parameters(self)
        return output

    def reset(self):
        super().reset()
        self.__emitter.reset()


class Emitter(abc.ABC, nn.Module):

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
        nn.Module.__init__(self)
        self.__prototype = prototype
        self.__delay = create_param(delay)
        self.__init_stored_mass = create_param(stored_mass)
        self.__stored_mass = 1 * self.__init_stored_mass
        self.__inital_time_passed = create_param(initial_time_passed)
        self.__time_since_last_emit = 1 * self.__inital_time_passed

    def __repr__(self) -> str:
        output = f"Emitter({repr(self.prototype)},{self.__delay.item()},"\
            + f"{self.__init_stored_mass.item()},"\
            + f"{self.__inital_time_passed.item()})"
        return output

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
            particle = self._create_particle()
            self.__stored_mass -= self.__prototype.mass
            return particle
        else:
            raise RuntimeError(
                "Cannot emit, delay not passed or too little mass stored")

    @abc.abstractmethod
    def _create_particle(self) -> EmittedMarble:
        pass

    def eat_mass(self, mass: float):
        self.__stored_mass = mass + self.__stored_mass

    def register_time_passed(self, time_passed: float):
        self.__time_since_last_emit += time_passed

    @property
    def delay(self):
        return self.__delay

    def set_delay(self, new_delay: nn.Parameter):
        """
        Set the learnable delay to equal (by reference!)
        the given delay. 
        """
        assert isinstance(new_delay, nn.Parameter), \
            "new_delay should be a torch.nn.Parameter"
        self.__delay = new_delay

    @property
    def init_time_passed(self):
        return self.__inital_time_passed

    def set_init_time_passed(self, new_time_passed: nn.Parameter):
        """
        Set the learnable init_time_passed to equal (by reference!)
        the given init_time_passed. 
        Also updates the current time since last emit to equal this value.
        """
        assert isinstance(new_time_passed, nn.Parameter), \
            "new_time_passed should be a torch.nn.Parameter"
        self.__inital_time_passed = new_time_passed
        self.__time_since_last_emit = 1 * new_time_passed

    @property
    def stored_mass(self):
        return self.__stored_mass

    @property
    def init_stored_mass(self):
        return self.__init_stored_mass

    def set_init_stored_mass(self, new_stored_mass: nn.Parameter):
        """
        Set the learnable init_stored_mass to equal (by reference!)
        the given stored mass. Also update the current stored mass
        to equal this amount (not by reference).
        """
        assert isinstance(new_stored_mass, nn.Parameter), \
            "new_stored_mass should be a torch.nn.Parameter"
        self.__init_stored_mass = new_stored_mass
        self.__stored_mass = 1 * self.__init_stored_mass

    def copy(self) -> Emitter:
        output = self.__class__(self.prototype,
                                self.delay,
                                self.stored_mass,
                                self.init_time_passed)
        output.set_delay(self.delay)
        output.set_init_time_passed(self.init_time_passed)
        output.set_init_stored_mass(self.init_stored_mass)
        return output

    def reset(self):
        """
        Reset the current time passed and the stored_mass
        back to their original values.
        """
        self.__time_since_last_emit = self.__inital_time_passed.clone()
        self.__stored_mass = self.__init_stored_mass.clone()


class MarbleEmitter(Emitter):

    def __init__(self,
                 prototype: Marble,
                 delay: float,
                 stored_mass: Optional[float] = 0,
                 initial_time_passed: Optional[float] = 0):
        if not isinstance(prototype, Marble):
            raise ValueError("Prototype of MarbleEmitter must be a Marble")
        super().__init__(prototype, delay, stored_mass, initial_time_passed)

    def __repr__(self) -> str:
        return "Marble" + super().__repr__()

    def _create_particle(self) -> EmittedMarble:
        p = self.prototype
        mass = (self.stored_mass/self.stored_mass.item())*self.prototype.mass
        output = EmittedMarble(p.pos, p.vel, p.acc, mass,
                               p._attraction_function, p.datum, 
                               p.marble_stiffness, p.node_stiffness, 
                               p.marble_attraction, p.node_attraction)
        return output


class MarbleEmitterVariablePosition(MarbleEmitter):
    """
    Subclass of MarbleEmitter that takes an optional
    new position for the emitted Marble in the method emit().
    """

    def __init__(self,
                 prototype: Marble,
                 delay: float,
                 stored_mass: Optional[float] = 0,
                 initial_time_passed: Optional[float] = 0,
                 rel_prototype_pos: torch.Tensor = torch.tensor([0.])):
        """
        Arguments:
        * Prototype: instance of object to be copied when emitting a new
            instance.
        * delay: amount of time between consecutive emits are allowed.
        * stored_mass: initial amount of mass stored, that is depleted when
            emitting.
        * initial_time_passed: how much time since the last delay is
            initially counted, used to set time until first real emit.
        * rel_prototype_pos: when emitting a new Marble,
            the new position of the emitter is given.
            This, plus the rel_prototype_pos, will become
            the position of the emitted Marble.
        """
        if rel_prototype_pos.shape != prototype.pos.shape:
            raise ValueError("relative prototype position different"
                             + "shape than prototype position")

        super().__init__(prototype, delay, stored_mass, initial_time_passed)
        self.__rel_prototype_pos = rel_prototype_pos

    @property
    def rel_prototype_pos(self) -> torch.Tensor:
        """
        Return the relative position to [new_pos] at which the Marble is
        emitted in .emit(new_pos),
        """
        return self.__rel_prototype_pos.clone()

    def emit(self, new_pos: Optional[torch.Tensor]) -> Node:
        output = super().emit()
        output.set_init_pos(nn.Parameter(new_pos + self.__rel_prototype_pos))
        output.pos = 1 * output.init_pos
        return output

    def __repr__(self) -> str:
        output = super().__repr__()
        output = re.sub("MarbleEmitter",
                        "MarbleEmitterVariablePosition", output)
        output = re.sub("\)$", f",{repr(self.__rel_prototype_pos)})", output)

        return output

    def copy(self) -> Emitter:
        output = self.__class__(self.prototype,
                                self.delay,
                                self.stored_mass,
                                self.init_time_passed,
                                self.__rel_prototype_pos)
        output.set_delay(self.delay)
        output.set_init_time_passed(self.init_time_passed)
        output.set_init_stored_mass(self.init_stored_mass)
        return output
