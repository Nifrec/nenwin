"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Base class for Nodes and Marbles, subclass of PhysicalParticle.
"""
from __future__ import annotations
import abc
import numpy as np
from typing import Union
from experiment_1.particle import PhysicalParticle


class Node(PhysicalParticle):
    """
    The particle that is used to build the architecture and represent data particles
    of the network:
    massfull 'nodes' that attract other nodes and/or marbles,
    and that way steer their path.
    They themselves may or may not be attracted by the marbles as well.
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
            on Marbles, in range [0, 1]. Overrides node_attaraction.
        * node_attaraction: multiplier for force exerted by self
            on non-Marble Nodes, in range [0, 1].
        """
        super().__init__(pos, vel, acc, mass, attraction_function)
        raise_error_if_any_not_in_range((marble_stiffness, node_stiffness,
                                         marble_attraction, node_attraction),
                                        lower=0,
                                        upper=1
                                        )
        self.__marble_stiffness = marble_stiffness
        self.__node_stiffness = node_stiffness
        self.__marble_attraction = marble_attraction
        self.__node_attraction = node_attraction

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

    def compute_attraction_force_to(
            self, other: Union[Marble, Node]) -> np.ndarray:
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
        return Node(self.pos,
                    self.vel,
                    self.acc,
                    self.mass,
                    self._attraction_function,
                    self.marble_stiffness,
                    self.node_stiffness,
                    self.marble_attraction,
                    self.node_attraction)

    def compute_experienced_force(self,
                                  other_particles: Set[Union[Marble, Node]]
                                  ) -> np.ndarray:
        """
        Given a set of other particles,
        find the resulting force this object experiences as excerted
        by the other particles. Keeps stiffness into account.
        """
        forces = np.zeros_like(self.acc)
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
            return self.__marble_stiffness
        elif isinstance(particle, Node):
            return self.__node_stiffness
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
                 pos: np.ndarray,
                 vel: np.ndarray,
                 acc: np.ndarray,
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

    @property
    def datum(self):
        return self.__datum

    def copy(self) -> Marble:
        return Marble(self.pos,
                      self.vel,
                      self.acc,
                      self.mass,
                      self._attraction_function,
                      self.datum,
                      self.marble_stiffness,
                      self.node_stiffness,
                      self.marble_attraction,
                      self.node_attraction)


def raise_error_if_any_not_in_range(values: Iterable[float],
                                    lower: float,
                                    upper: float):
    if any(x < lower or x > upper for x in values):
        raise ValueError("Expected value in range[0, 1]")
