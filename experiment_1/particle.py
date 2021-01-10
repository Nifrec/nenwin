"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

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

Most fundamental pieces for building Nenwin-networks.
"""
from __future__ import annotations
import abc
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union

from experiment_1.constants import DEVICE


class Particle(abc.ABC, nn.Module):
    """
    Abstract Base Class for all particles of the simulation,
    both the data-particles and the nodes of the model itself.
    """

    def __init__(self,
                 pos: torch.Tensor,
                 vel: torch.Tensor,
                 acc: torch.Tensor,
                 device: Optional[Union[torch.device, str]] = DEVICE):
        nn.Module.__init__(self)
        self.__device = device
        self.__check_input_dims(pos, vel, acc)
        self.__pos = create_vector_param(pos, device)
        self.__prev_pos = torch.tensor(pos, dtype=torch.float, device=device)
        self.__vel = create_vector_param(vel, device)
        self.__acc = create_vector_param(acc, device)
        # Previous value of self.acc (updated when self.acc changes)
        self._prev_acc = torch.tensor(acc, dtype=torch.float, device=device)
        # The value of self.acc that came *before* prev_acc
        self._prev_prev_acc = torch.tensor(
            acc, dtype=torch.float, device=device)

    def __check_input_dims(self, pos, vel, acc):
        if (pos.shape != vel.shape) or (pos.shape != acc.shape):
            raise ValueError("Input values have mismatching dimensions: "
                             + f"{pos.shape}, {vel.shape}, {acc.shape}")

    @property
    def device(self) -> torch.device:
        return self.__device

    @ property
    def pos(self) -> torch.Tensor:
        return self.__pos.clone().detach().requires_grad_(False)

    @ property
    def vel(self) -> torch.Tensor:
        return self.__vel.clone().detach().requires_grad_(False)

    @ property
    def acc(self) -> torch.Tensor:
        return self.__acc.clone().detach().requires_grad_(False)

    @ pos.setter
    def pos(self, new_pos: torch.Tensor):
        if (new_pos.shape != self.__pos.shape):
            raise RuntimeError("New position particle has different dimension")
        new_pos = create_vector_param(new_pos, self.device)
        self.__pos = new_pos

    @ acc.setter
    def acc(self, new_acc: torch.Tensor):
        if (new_acc.shape != self.__acc.shape):
            raise RuntimeError(
                "New acceleration particle has different dimension")
        self._set_prev_accs()
        new_acc = create_vector_param(new_acc, self.device)
        self.__acc = new_acc

    def _set_prev_accs(self):
        """
        Sets current value of self.acc as the previous value,
        and updates self._prev_acc and self._prev_prev_acc accordingly.
        """
        self._prev_prev_acc = self._prev_acc
        self._prev_acc = self.acc

    @ vel.setter
    def vel(self, new_vel: torch.Tensor):
        if (new_vel.shape != self.__vel.shape):
            raise RuntimeError("New velocity particle has different dimension")
        new_vel = create_vector_param(new_vel, self.device)
        self.__vel = new_vel

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


class PhysicalParticle(Particle):
    """
    Particle with mass (negative mass allowed).
    Has a gravity field defines around it, that can locally be inferred.
    """

    def __init__(self,
                 pos: torch.Tensor,
                 vel: torch.Tensor,
                 acc: torch.Tensor,
                 mass: float,
                 attraction_function: callable):
        super().__init__(pos, vel, acc)
        self._attraction_function = attraction_function
        self.__mass = mass

    @ property
    def mass(self) -> float:
        return self.__mass

    @ mass.setter
    def mass(self, new_mass: float):
        self.__mass = new_mass

    def update_acceleration(self, forces: torch.Tensor):
        """
        Updates the acceleration of this particle according to Newton's
        Second Law (F_res = mass * acc).

        Expected format of forces:
        A 2-dimensional array, the first dimension indexing the individual
        forces, and the second dimension indexing the values of force vectors
        themselves.
        """
        if forces.size == 0:
            self.acc = np.zeros_like(self.acc, dtype=float)
            return
        if (len(forces.shape) != 2) or (forces[0].shape != self.acc.shape):
            raise ValueError(
                "Unexpected shape of forces-array, expected 2 dims")
        self._set_prev_accs()
        self.acc = np.sum(forces, axis=0) / abs(self.mass)

    def compute_attraction_force_to(
            self, other: PhysicalParticle) -> torch.Tensor:
        """
        Computes the force vector induced by this particle to the
        [other] paricle at the position of the other particle.
        """
        difference_vector = (self.pos - other.pos)
        radius = np.linalg.norm(difference_vector)
        direction = difference_vector / radius

        return direction * self._attraction_function(self, other)

    def copy(self) -> PhysicalParticle:
        return PhysicalParticle(self.pos, self.vel, self.acc, self.mass,
                                self._attraction_function)


def create_vector_param(vector: Union[np.ndarray, torch.Tensor],
        device: torch.device = DEVICE) -> nn.Parameter:
        output = torch.tensor(vector, dtype=torch.float, device=device)
        return nn.Parameter(output)
