"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Class representing the state of a simulation:
    keeps track of the nodes, marbles, and advances timesteps.
Also related auxiliary and communication classes.
"""
import numpy as np
from typing import Set, Iterable, Optional, List
import multiprocessing
import multiprocessing.connection
import enum
from numbers import Number
import warnings
import abc

from experiment_1.particle import PhysicalParticle
from experiment_1.stiffness_particle import Node
from experiment_1.marble_eater_node import MarbleEaterNode
from experiment_1.stiffness_particle import Marble
import experiment_1.aux as aux


class UICommands(enum.Enum):
    """
    Commands that the UI can give to a running NenwinModel.
    """
    # Stop/pause the simulation.
    stop = "stop"
    # Add a new input to the simulation
    # (should come together with new input values).
    read_input = "input"
    # Write current output values to the pipe.
    write_output = "output"


class UIMessage():
    def __init__(self, command: UICommands, data: Optional[object] = None):
        self.command = command
        self.data = data


class NenwinModel():
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
        self.__nodes = set(nodes)
        self.__eater_nodes: List[MarbleEaterNode] =\
            [node for node in nodes if isinstance(node, MarbleEaterNode)]
        if initial_marbles is not Node:
            self.__marbles = set(initial_marbles)
        else:
            initial_marbles = set()
        self.__all_particles = self.__nodes.union(self.__marbles)

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
        self.__marbles.update(new_marbles)

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

    def __compute_net_force_for(self, particle: Node) -> np.ndarray:
        """
        Compute net force for [particle] as excerted on it by 
        all other particles in the simulation.
        The result is a 2D matrix with a single row.
        """
        other_particles = self.__all_particles.difference((particle,))
        net_force = np.zeros((1,) + particle.acc.shape)
        net_force += particle.compute_experienced_force(other_particles)
        return net_force

    def __feed_marble_if_close_to_any_eater(self, marble: Marble):
        for eater in self.__eater_nodes:
            if aux.distance(marble, eater) <= eater.radius:
                eater.eat(marble)
                self.__marbles.remove(marble)
                break


# class NumMarblesEatenAsOutputModel(NenwinModel):
#     """
#     Variant of the Nenwin model that outputs the integer amount of
#     Marbles that fel in each MarbleEaterNode, collected in an array,
#     as output.
#     """

#     def _produce_outputs(self) -> np.ndarray:
#         outputs = np.empty((len(self.marble_eater_nodes)))
#         for idx in range(len(self.marble_eater_nodes)):
#             outputs[idx] = self.marble_eater_nodes[idx].num_marbles_eaten
#         return outputs
