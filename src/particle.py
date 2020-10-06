"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée

Most fundamental pieces for building Nenwin-networks.
"""
import abc
import torch


class Particle(abc.ABC):
    """
    Abstract Base Class for all particles of the simulation,
    both the data-particles and the nodes of the model itself.
    """

    def __init__(self,
                 pos: torch.Tensor,
                 velocity: torch.Tensor,
                 acceleration: torch.Tensor):
                 
        self.__check_input_dims(pos, velocity, acceleration)
        self.__pos = pos
        self.__vel = velocity
        self.__acc = acceleration

    def __check_input_dims(self, pos, vel, acc):
        if (pos.shape != vel.shape) or (pos.shape != acc.shape):
            raise ValueError("Input values have mismatching dimensions")

    @property
    def pos():
        return self.__pos

    @property
    def vel():
        return self.__vel

    @property
    def acc():
        return self.__acc

    @pos.setter
    def pos(new_pos: torch.Tensor):
        if (new_pos.shape != self.__pos.shape):
            raise RuntimeError("New position particle has different dimension")
        self.__pos = new_pos

    @acc.setter
    def acc(new_acc: torch.Tensor):
        if (new_acc.shape != self.__acc.shape):
            raise RuntimeError(
                "New acceleration particle has different dimension")
        self.__acc = new_acc

    @vel.setter
    def vel(new_vel: torch.Tensor):
        if (new_vel.shape != self.__vel.shape):
            raise RuntimeError("New velocity particle has different dimension")
        self.__vel = new_vel

    @abc.abstractmethod
    def update(time_passed: float):
        pass
