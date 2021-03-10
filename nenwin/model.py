"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
March 2021

Copyright (C) 2021 Lulof Pirée, Teun Schilperoort

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

Class representing the state of a simulation:
    keeps track of the nodes, marbles, and advances timesteps.
"""
import torch
import torch.nn as nn
from typing import Set, Iterable, Optional, List

from nenwin.node import Node
from nenwin.marble_eater_node import MarbleEaterNode
from nenwin.node import Marble
import nenwin.auxliary as auxliary

class NenwinModel(nn.Module):
    """
    Class representing the state of a simulation:
    keeps track of the nodes, marbles, and advances timesteps.
    """

    def __init__(self,
                 nodes: Iterable[Node],
                 initial_marbles: Optional[Iterable[Marble]] = None):
        """
        Set up the model, given a number of Node instances,
        and a step-size for the simulation.
        Optionally some input Marble's can already be added,
        which will move as soon as NenwinModel.run() is called.
        """
        super().__init__()
        self.__nodes = set(nodes)
        self.__eater_nodes: List[MarbleEaterNode] =\
            [node for node in nodes if isinstance(node, MarbleEaterNode)]
        if initial_marbles is not None:
            self.__marbles = set(initial_marbles)
        else:
            self.__marbles = set([])
        self.__all_particles = self.__nodes.union(self.__marbles)

        self.__register_particles_as_modules(self.__all_particles)

    def __register_particles_as_modules(self, particles: Iterable[Node]):
        """
        Particles are stored in sets, but this way they are not registered
        as submodules within the PyTorch framework.
        By registering them as modules it becomes possible to get
        all trainable parameters at once via the NenwinModel.
        """
        for particle in particles:
            self.add_module(str(id(particle)), particle)

    def __repr__(self) -> str:
        return f"NenwinModel({repr(self.__nodes)},{repr(self.__marbles)})"

    @property
    def nodes(self) -> Set[Node]:
        """
        Get the set of all currently simulated nodes,
        each in their current state.
        """
        return self.__nodes.copy()

    @property
    def marble_eater_nodes(self) -> List[MarbleEaterNode]:
        """
        Get all the MarbleEaterNode's present in the nodes of this NenwinModel.
        They are returned in the same order as their respective output.
        """
        return self.__eater_nodes.copy()

    @property
    def marbles(self) -> Set[Marble]:
        """
        Get the set of all currently simulated marbles,
        each in their current state.
        """
        return self.__marbles.copy()

    def add_marbles(self, new_marbles: Iterable[Marble]):
        """
        Add more Marbles to the set of actively simulated Marbles.
        """
        self.__register_particles_as_modules(new_marbles)
        self.__marbles.update(new_marbles)
        self.__all_particles.update(new_marbles)

    def make_timestep(self, time_passed: float):
        """
        Advance the simulation [time_passed] time
        (i.e. update position, velocity, acceleration of particles,
        and consumption by Eater-Nodes).
        """
        for particle in self.__all_particles:
            net_force = self.__compute_net_force_for(particle)
            particle.update_acceleration(net_force)

            for particle in self.__all_particles:
                particle.update_movement(time_passed)

            for marble in list(self.__marbles):
                self.__feed_marble_if_close_to_any_eater(marble)

    def __compute_net_force_for(self, particle: Node) -> torch.Tensor:
        """
        Compute net force for [particle] as excerted on it by 
        all other particles in the simulation.
        The result is a 2D matrix with a single row.
        """
        other_particles = self.__all_particles.difference((particle,))
        net_force = torch.zeros((1,) + particle.acc.shape)
        net_force += particle.compute_experienced_force(other_particles)
        return net_force

    def __feed_marble_if_close_to_any_eater(self, marble: Marble):
        for eater in self.__eater_nodes:
            if auxliary.distance(marble, eater) <= eater.radius:
                eater.eat(marble)
                self.__marbles.remove(marble)
                break