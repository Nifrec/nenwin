"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Most fundamental pieces for building Nenwin-networks.
"""
from __future__ import annotations
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
        self.__pos = pos.astype(np.float)
        self.__prev_pos = self.__pos
        self.__vel = velocity.astype(np.float)
        self.__acc = acceleration.astype(np.float)
        # Previous value of self.acc (updated when self.acc changes)
        self._prev_acc = self.__acc
        # The value of self.acc that came *before* prev_acc
        self._prev_prev_acc = self.__acc

    def __check_input_dims(self, pos, vel, acc):
        if (pos.shape != vel.shape) or (pos.shape != acc.shape):
            raise ValueError("Input values have mismatching dimensions: "
                             + f"{pos.shape}, {vel.shape}, {acc.shape}")

    @property
    def pos(self) -> np.ndarray:
        return self.__pos.copy()

    @property
    def vel(self) -> np.ndarray:
        return self.__vel.copy()

    @property
    def acc(self) -> np.ndarray:
        return self.__acc.copy()

    @pos.setter
    def pos(self, new_pos: np.ndarray):
        if (new_pos.shape != self.__pos.shape):
            raise RuntimeError("New position particle has different dimension")
        np.copyto(self.__pos, new_pos)

    @acc.setter
    def acc(self, new_acc: np.ndarray):
        if (new_acc.shape != self.__acc.shape):
            raise RuntimeError(
                "New acceleration particle has different dimension")
        self._set_prev_accs()
        np.copyto(self.__acc, new_acc)

    def _set_prev_accs(self):
        """
        Sets current value of self.acc as the previous value,
        and updates self._prev_acc and self._prev_prev_acc accordingly.
        """
        self._prev_prev_acc = self._prev_acc
        self._prev_acc = self.acc

    @vel.setter
    def vel(self, new_vel: np.ndarray):
        if (new_vel.shape != self.__vel.shape):
            raise RuntimeError("New velocity particle has different dimension")
        np.copyto(self.__vel, new_vel)

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


class PhysicalParticle(Particle):
    """
    Particle with mass (negative mass allowed).
    Has a gravity field defines around it, that can locally be inferred.
    """

    def __init__(self,
                 pos: np.ndarray,
                 vel: np.ndarray,
                 acc: np.ndarray,
                 mass: float,
                 attraction_function: callable):
        super().__init__(pos, vel, acc)
        self._attraction_function = attraction_function
        self.__mass = mass

    @property
    def mass(self) -> float:
        return self.__mass

    @mass.setter
    def mass(self, new_mass: float):
        self.__mass = new_mass

    def update_acceleration(self, forces: np.ndarray):
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
            self, other: PhysicalParticle) -> np.ndarray:
        """
        Computes the force vector induced by this particle to the
        [other] paricle at the position of the other particle.
        """
        difference_vector = (self.pos - other.pos)
        radius = np.linalg.norm(difference_vector)
        direction = difference_vector / radius

        return direction * self._attraction_function(self, other)
