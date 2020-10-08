"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée

Most fundamental pieces for building Nenwin-networks.
"""
import abc
import numpy as np


class Particle(abc.ABC):
    """
    Abstract Base Class for all particles of the simulation,
    both the data-particles and the nodes of the model itself.
    """

    def __init__(self,
                 pos: np.ndarray,
                 velocity: np.ndarray,
                 acceleration: np.ndarray):

        self.__check_input_dims(pos, velocity, acceleration)
        self.__pos = pos
        self.__vel = velocity
        self.__acc = acceleration

    def __check_input_dims(self, pos, vel, acc):
        if (pos.shape != vel.shape) or (pos.shape != acc.shape):
            raise ValueError("Input values have mismatching dimensions: "
                             + f"{pos.shape}, {vel.shape}, {acc.shape}")

    @property
    def pos(self) -> np.ndarray:
        return self.__pos

    @property
    def vel(self) -> np.ndarray:
        return self.__vel

    @property
    def acc(self) -> np.ndarray:
        return self.__acc

    @pos.setter
    def pos(self, new_pos: np.ndarray):
        if (new_pos.shape != self.__pos.shape):
            raise RuntimeError("New position particle has different dimension")
        self.__pos = new_pos

    @acc.setter
    def acc(self, new_acc: np.ndarray):
        if (new_acc.shape != self.__acc.shape):
            raise RuntimeError(
                "New acceleration particle has different dimension")
        self.__acc = new_acc

    @vel.setter
    def vel(self, new_vel: np.ndarray):
        if (new_vel.shape != self.__vel.shape):
            raise RuntimeError("New velocity particle has different dimension")
        self.__vel = new_vel

    @abc.abstractmethod
    def update(self, time_passed: float):
        pass


class PhysicalParticle(Particle):
    """
    Particle with mass (negative mass allowed).
    Has a gravity field defines around it, that can locally be inferred.
    """

    def __init__(self,
                 pos: np.ndarray,
                 vel: np.ndarray,
                 acc: np.ndarray,
                 mass: np.ndarray):
        super().__init__(pos, vel, acc)

    def update(self, time_passed: float):
        self.vel += time_passed*self.acc
        self.pos += time_passed*self.pos
