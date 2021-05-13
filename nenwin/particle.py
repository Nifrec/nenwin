"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
May 2021

Copyright (C) 2021 Lulof Pirée, 

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

Most fundamental pieces for building Nenwin-networks.
"""
from __future__ import annotations
import abc
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union

from nenwin.constants import DEVICE


class Particle(abc.ABC, nn.Module):
    """
    Abstract Base Class for all particles of the simulation,
    both the data-particles and the nodes of the model itself.
    """

    def __init__(self,
                 pos: Union[np.ndarray, torch.Tensor],
                 vel: Union[np.ndarray, torch.Tensor],
                 acc: Union[np.ndarray, torch.Tensor],
                 device: Optional[Union[torch.device, str]] = DEVICE):
        nn.Module.__init__(self)
        self.__device = device
        self.__check_input_dims(pos, vel, acc)

        self.__pos = create_param(pos, device, only_tensor=True)
        self.__vel = create_param(vel, device, only_tensor=True)
        self.__acc = create_param(acc, device, only_tensor=True)

        # Previous value of self.acc (updated when self.acc changes)
        self._prev_acc = self.__init_prev_value(acc)
        # The value of self.acc that came *before* prev_acc
        self._prev_prev_acc = self.__init_prev_value(acc)

    def __check_input_dims(self, pos, vel, acc):
        if (pos.shape != vel.shape) or (pos.shape != acc.shape):
            raise ValueError("Input values have mismatching dimensions: "
                             + f"{pos.shape}, {vel.shape}, {acc.shape}")

    def __init_prev_value(self,
                          vector: Union[np.ndarray, torch.Tensor]
                          ) -> torch.Tensor:
        """
        Create correct datastructure for a non-trainable vector variable such as
        self._prev_acc, self._prev_prev_acc
        """
        if isinstance(vector, torch.Tensor):
            return vector.clone().detach()
        else:
            return torch.tensor(vector, dtype=torch.float,
                                device=self.device)

    def __repr__(self) -> str:
        pos_str = repr(self.pos.detach())
        vel_str = repr(self.vel.detach())
        acc_str = repr(self.acc.detach())
        output = f"Particle({pos_str},{vel_str},{acc_str},"
        output += f"{repr(self.device)})"
        return output

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def pos(self) -> torch.Tensor:
        return self.__pos.clone()

    @property
    def vel(self) -> torch.Tensor:
        return self.__vel.clone()

    @property
    def acc(self) -> torch.Tensor:
        return self.__acc.clone()

    @pos.setter
    def pos(self, new_pos: torch.Tensor):
        if (new_pos.shape != self.__pos.shape):
            raise RuntimeError("New position particle has different dimension")
        self.__pos = create_param(new_pos, self.device, only_tensor=True)

    @vel.setter
    def vel(self, new_vel: torch.Tensor):
        if (new_vel.shape != self.__vel.shape):
            raise RuntimeError("New velocity particle has different dimension")
        self.__vel = create_param(new_vel, self.device, only_tensor=True)

    @acc.setter
    def acc(self, new_acc: torch.Tensor):
        if (new_acc.shape != self.__acc.shape):
            raise RuntimeError(
                "New acceleration particle has different dimension")
        self._set_prev_accs()
        self.__acc = create_param(new_acc, self.device, only_tensor=True)

    def _set_prev_accs(self):
        """
        Sets current value of self.acc as the previous value,
        and updates self._prev_acc and self._prev_prev_acc accordingly.
        """
        self._prev_prev_acc = self._prev_acc
        self._prev_acc = self.acc

    def update_movement(self, time_passed: float):
        """
        Update the movement of the current particle according
        to Newtonian mechanics, for a period of time [time_passed].
        Uses constant acceleration.

        Uses the Beeman-algorithm to make the numerical integration
        of the laws of motion.
        https://en.wikipedia.org/wiki/Beeman%27s_algorithm#cite_note-beeman76-2
        """
        self.pos = self.pos + time_passed * self.vel + (1/6)*(time_passed**2)*(
            4*self._prev_acc - self._prev_prev_acc)
        self.vel = self.vel + (1/12)*time_passed*(
            5*self.acc + 8*self._prev_acc - self._prev_prev_acc)

    def copy(self) -> Particle:
        return Particle(self.pos, self.vel, self.acc)


class InitialValueParticle(Particle):
    """
    Variant of Particle with the same properties,
    but with its *initial* pos, vel and acc as trainable parameters.
    Has a function to restore its pos, vel and acc 
    to the original (optimized) value.
    """

    def __init__(self,
                 pos: Union[np.ndarray, torch.Tensor],
                 vel: Union[np.ndarray, torch.Tensor],
                 acc: Union[np.ndarray, torch.Tensor],
                 device: Optional[Union[torch.device, str]] = DEVICE):
        Particle.__init__(self, pos, vel, acc, device)
        self.__init_pos = create_param(pos, device)
        self.__init_vel = create_param(vel, device)
        self.__init_acc = create_param(acc, device)

        # Note: detach() is intentionally *not* used,
        # the initial values need to be optimized
        self.pos = 1 * self.__init_pos
        self.vel = 1 * self.__init_vel
        self.acc = 1 * self.__init_acc

    def __repr__(self) -> str:

        pos_as_str = repr(self.init_pos.clone().detach())
        vel_as_str = repr(self.init_vel.clone().detach())
        acc_as_str = repr(self.init_acc.clone().detach())

        output = f"InitialValueParticle({pos_as_str}"
        output += f",{vel_as_str},{acc_as_str},"
        output += f"{repr(self.device)})"
        return output

    @property
    def init_pos(self):
        return self.__init_pos

    @property
    def init_vel(self):
        return self.__init_vel

    @property
    def init_acc(self):
        return self.__init_acc

    def set_init_pos(self, new_init_pos: nn.Parameter):
        """
        Directly change the init_pos *by reference*.
        This is the only way to ensure compuational graphs will also be set.
        """
        if not isinstance(new_init_pos, nn.Parameter):
            raise ValueError("Initial position should be a torch.nn.Paratemer")
        self.__init_pos = new_init_pos

    def set_init_vel(self, new_init_vel: nn.Parameter):
        """
        Directly change the init_vel *by reference*.
        This is the only way to ensure compuational graphs will also be set.
        """
        if not isinstance(new_init_vel, nn.Parameter):
            raise ValueError("Initial velition should be a torch.nn.Paratemer")
        self.__init_vel = new_init_vel

    def set_init_acc(self, new_init_acc: nn.Parameter):
        """
        Directly change the init_acc *by reference*.
        This is the only way to ensure compuational graphs will also be set.
        """
        if not isinstance(new_init_acc, nn.Parameter):
            raise ValueError("Initial accition should be a torch.nn.Paratemer")
        self.__init_acc = new_init_acc

    def reset(self):
        self.pos = self.__init_pos.clone()
        self.vel = self.__init_vel.clone()
        self.acc = self.__init_acc.clone()

        self._prev_acc = self.init_acc.clone()
        self._prev_prev_acc = self._prev_acc.clone()

    def copy(self) -> InitialValueParticle:
        output = InitialValueParticle(self.init_pos, self.init_vel,
                                      self.init_acc)
        output.adopt_parameters(self)

        return output

    def adopt_parameters(self, source: InitialValueParticle):
        """
        Sets the initial position, velocity and acceleration
        to the same object (so by reference) to the init pos, vel and acc
        of [source].
        """
        self.set_init_pos(source.init_pos)
        self.set_init_vel(source.init_vel)
        self.set_init_acc(source.init_acc)


class PhysicalParticle(InitialValueParticle):
    """
    Particle with mass (negative mass allowed).
    Has a gravity field defines around it, that can locally be inferred.
    """

    def __init__(self,
                 pos: torch.Tensor,
                 vel: torch.Tensor,
                 acc: torch.Tensor,
                 mass: float,
                 attraction_function: callable,
                 device: Optional[Union[torch.device, str]] = DEVICE):
        super().__init__(pos, vel, acc, device)
        self._attraction_function = attraction_function
        self.__mass = create_param(mass, device=self.device)

    @property
    def mass(self) -> torch.Tensor:
        return self.__mass

    def set_mass(self, new_mass):
        """
        Directly change the mass *by reference*.
        This is the only way to ensure compuational graphs will also be set.
        """
        if not isinstance(new_mass, nn.Parameter):
            raise ValueError("New mass should be a torch.nn.Paratemer")
        self.__mass = new_mass

    def __repr__(self) -> str:
        output = super().__repr__()
        output = output.replace("InitialValue", "Physical")
        output = output.replace(",device",
                                f",{self.mass.item()},"
                                + f"{repr(self._attraction_function)},device")
        return output

    def update_acceleration(self, forces: torch.Tensor):
        """
        Updates the acceleration of this particle according to Newton's
        Second Law (F_res = mass * acc).

        Expected format of forces:
        A 2-dimensional array, the first dimension indexing the individual
        forces, and the second dimension indexing the values of force vectors
        themselves.
        """
        if forces.shape[0] == 0:
            self.acc = 0 * self.acc
            return
        if (len(forces.shape) != 2) or (forces[0].shape != self.acc.shape):
            raise ValueError(
                "Unexpected shape of forces-array, expected 2 dims")
        self._set_prev_accs()
        self.acc = torch.sum(forces, dim=0) / torch.abs(self.mass)

    def compute_attraction_force_to(
            self, other: PhysicalParticle) -> torch.Tensor:
        """
        Computes the force vector induced by this particle to the
        [other] paricle at the position of the other particle.
        """
        difference_vector = (self.pos - other.pos)
        radius = torch.norm(difference_vector)
        direction = difference_vector / radius

        return direction * self._attraction_function(self, other)

    def copy(self) -> PhysicalParticle:
        output = PhysicalParticle(self.init_pos,
                                  self.init_vel,
                                  self.init_acc,
                                  self.mass,
                                  self._attraction_function)
        output.adopt_parameters(self)
        return output

    def adopt_parameters(self, source: PhysicalParticle):
        super().adopt_parameters(source)
        self.set_mass(source.mass)


def create_param(vector: Union[np.ndarray, torch.Tensor, float],
                 device: torch.device = DEVICE,
                 only_tensor: bool = False) -> nn.Parameter:
    """
    Convert a vector (either torch.Tensor or np.ndarray) 
    to the right format to be an attribute of a Particle.
    By default converted to a torch.nn.Parameter.
    Does *NOT* preserve grad and grad_fn.
    """
    if isinstance(vector, torch.Tensor):
        output = vector.clone()
    else:
        output = torch.tensor(vector,
                              dtype=torch.float,
                              device=device,
                              requires_grad=True)
    if only_tensor:
        return output
    else:
        return nn.Parameter(output)
