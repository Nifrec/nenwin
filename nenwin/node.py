"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
March 2021

Copyright (C) 2020 Lulof Pirée, Teun Schilperoort

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

Base class for Nodes and Marbles, subclass of PhysicalParticle.
"""
from __future__ import annotations
import abc
import numpy as np
import torch
import torch.nn as nn
import re
from typing import Iterable, Set, Union
from nenwin.particle import PhysicalParticle


class Node(PhysicalParticle):
    """
    The particle that is used to build the architecture and represent data particles
    of the network:
    massfull 'nodes' that attract other nodes and/or marbles,
    and that way steer their path.
    They themselves may or may not be attracted by the marbles as well.
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
                 node_attraction: float):
        """
        Raises error if pos, vel and/or acc vectors do not have the same shape.

        Arguments:
        * pos: vector describing initial position of Node.
        * vel: vector describing initial velocity of Node.
        * acc: vector describing initial acceleration of Node.
        * mass: attraction strength of Node, negative values are allowed.
        * attraction_function: function taking two particles and computes
            attraction force exerted by first (this Node) on second.
        * marble_stiffness: multiplier for force exerted on this
            by other Marbles, in range [0, 1]. Overrides node_stiffness.
        * marble_stiffness: multiplier for force exerted on this
            by other non-Marble Nodes, in range [0, 1].
        * marble_attraction: multiplier for force exerted by self
            on Marbles, in range [0, 1]. Overrides node_attraction.
        * node_attaraction: multiplier for force exerted by self
            on non-Marble Nodes, in range [0, 1].
        """
        super().__init__(pos, vel, acc, mass, attraction_function)
        raise_error_if_any_not_in_range((marble_stiffness, node_stiffness,
                                         marble_attraction, node_attraction),
                                        lower=0,
                                        upper=1
                                        )
        self.__marble_stiffness = self.__make_parameter(marble_stiffness)
        self.__node_stiffness = self.__make_parameter(node_stiffness)
        self.__marble_attraction = self.__make_parameter(marble_attraction)
        self.__node_attraction = self.__make_parameter(node_attraction)

    def __make_parameter(self, value: float) -> nn.Parameter:
        value_as_tensor = torch.tensor(value,
                                       dtype=torch.float,
                                       device=self.device)
        return nn.Parameter(value_as_tensor)

    def __repr__(self) -> str:
        output = super().__repr__()
        output = output.replace("PhysicalParticle", "Node")
        output = re.sub(r',device\(.*\)', '', output)
        output += f",{self.marble_stiffness}"
        output += f",{self.node_stiffness}"
        output += f",{self.marble_attraction}"
        output += f",{self.node_attraction})"
        return output

    @property
    def marble_stiffness(self) -> float:
        return self.__marble_stiffness

    @property
    def node_stiffness(self) -> float:
        return self.__node_stiffness

    @property
    def marble_attraction(self) -> float:
        return self.__marble_attraction

    @property
    def node_attraction(self) -> float:
        return self.__node_attraction

    def set_marble_stiffness(self, new_stiffness: nn.Parameter):
        self.__marble_stiffness = new_stiffness

    def set_node_stiffness(self, new_stiffness: nn.Parameter):
        self.__node_stiffness = new_stiffness

    def set_marble_attraction(self, new_attraction: nn.Parameter):
        self.__marble_attraction = new_attraction

    def set_node_attraction(self, new_attraction: nn.Parameter):
        self.__node_attraction = new_attraction

    def compute_attraction_force_to(
            self, other: Union[Marble, Node]) -> torch.Tensor:
        """
        Computes the force vector induced by this particle to the
        [other] paricle at the position of the other particle.
        """
        if isinstance(other, Marble):
            multiplier = self.marble_attraction
        elif isinstance(other, Node):
            multiplier = self.node_attraction
        else:
            raise ValueError("Expected Node or Marble as [other]")

        return multiplier * super().compute_attraction_force_to(other)

    def copy(self) -> Node:
        output = Node(self.init_pos,
                      self.init_vel,
                      self.init_acc,
                      self.mass,
                      self._attraction_function,
                      self.marble_stiffness,
                      self.node_stiffness,
                      self.marble_attraction,
                      self.node_attraction)
        output.adopt_parameters(self)
        return output

    def adopt_parameters(self, source: Node):
        super().adopt_parameters(source)
        self.set_marble_stiffness(source.marble_stiffness)
        self.set_node_stiffness(source.node_stiffness)
        self.set_marble_attraction(source.marble_attraction)
        self.set_node_attraction(source.node_attraction)

    def compute_experienced_force(self,
                                  other_particles: Set[Union[Marble, Node]]
                                  ) -> torch.Tensor:
        """
        Given a set of other particles,
        find the resulting force this object experiences as excerted
        by the other particles. Keeps stiffness into account.
        """
        forces = torch.zeros_like(self.acc, requires_grad=False)
        for particle in other_particles:
            stiffness = self.__find_stiffness_to(particle)
            forces += (1-stiffness) * \
                particle.compute_attraction_force_to(self)

        return forces

    def __find_stiffness_to(self, particle: Union[Marble, Node]) -> float:
        """
        Throws error if the particle is neither a Node or a Marble.
        """
        if isinstance(particle, Marble):
            return self.marble_stiffness
        elif isinstance(particle, Node):
            return self.node_stiffness
        else:
            raise ValueError(
                "__find_stiffness_to: particle is neither Node nor Marble")


class Marble(Node):
    """
    The particle that is used to build the architecture of the network:
    massfull 'nodes' that attract the marbles, and that way steer their path.
    They themselves may or may not be attracted by the marbles as well.
    """

    def __init__(self,
                 pos: torch.Tensor,
                 vel: torch.Tensor,
                 acc: torch.Tensor,
                 mass: float,
                 attraction_function: callable,
                 datum: object,
                 marble_stiffness: float = 1,
                 node_stiffness: float = 0,
                 marble_attraction: float = 0,
                 node_attraction: float = 1):
        """
        Initialize the Marble at the given position with the given acceleration,
        velocity and mass. The Marble's attraction to any other Particle
        is described by the[attraction_function].

        The datum is the object that the Marble represent.
        What form it may have is not specified.
        """
        super().__init__(pos, vel, acc, mass, attraction_function,
                         marble_stiffness, node_stiffness, marble_attraction, node_attraction)
        self.__datum = datum

    def __repr__(self):
        output = super().__repr__()
        output = output.replace("Node", "Marble")
        output = re.sub(r'(([0-9]\.[0-9]*,?){4})', f'{self.datum},\\1', output)
        return output

    @property
    def datum(self):
        return self.__datum

    def copy(self) -> Marble:
        output = Marble(self.init_pos,
                        self.init_vel,
                        self.init_acc,
                        self.mass,
                        self._attraction_function,
                        self.datum,
                        self.marble_stiffness,
                        self.node_stiffness,
                        self.marble_attraction,
                        self.node_attraction)
        output.adopt_parameters(self)
        return output


class EmittedMarble(Marble):
    """
    Identical to Marble, except that the .mass attribute is not a nn.Parameter
    and hence a leaf (in any computational graph used by autograd),
    but a non-leaf normal tensor.
    """

    def __init__(self,
                 pos: torch.Tensor,
                 vel: torch.Tensor,
                 acc: torch.Tensor,
                 mass: torch.Tensor,
                 attraction_function: callable,
                 datum: object,
                 marble_stiffness: float,
                 node_stiffness: float,
                 marble_attraction: float,
                 node_attraction: float):
        super().__init__(pos, vel, acc, mass,
                         attraction_function,
                         datum,
                         marble_stiffness=marble_stiffness,
                         node_stiffness=node_stiffness,
                         marble_attraction=marble_attraction,
                         node_attraction=node_attraction)
        if not isinstance(mass, torch.Tensor):
            raise ValueError("Initializing an EmittedMarble "
                             + "whose mass is not a tensor.")
        if mass.is_leaf:
            raise ValueError("Initializing an EmittedMarble with a leaf mass")

        del self._PhysicalParticle__mass
        self.__mass = mass.clone()

    def copy(self):
        raise RuntimeError("EmittedMarble's should never be copied.")

    def set_mass(self, new_mass) -> torch.Tensor:
        self.__mass = new_mass

    @property
    def mass(self) -> torch.Tensor:
        return self.__mass

    def __repr__(self):
        raise RuntimeError("EmittedMarble's should never be __repr__'ed.")


def raise_error_if_any_not_in_range(values: Iterable[float],
                                    lower: float,
                                    upper: float):
    if any(x < lower or x > upper for x in values):
        raise ValueError("Expected value in range[0, 1]")
