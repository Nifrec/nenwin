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
        self.__prev_acc = self.__acc
        self.__prev_prev_acc = self.__acc

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
        np.copyto(self.__acc, new_acc)

    @vel.setter
    def vel(self, new_vel: np.ndarray):
        if (new_vel.shape != self.__vel.shape):
            raise RuntimeError("New velocity particle has different dimension")
        np.copyto(self.__vel, new_vel)

    def update(self, time_passed: float):
        """
        Update the movement of the current particle according
        to Newtonian mechanics, for a period of time [time_passed].
        Uses constant acceleration.

        Uses the Verlet-algorithm to make the numerical integration
        of the laws of motion.
        (sources: 
            https://www.compadre.org/PICUP/resources/Numerical-Integration/
            https://www.algorithm-archive.org/contents/verlet_integration/verlet_integration.html
            )
        """
        # current_pos = self.pos # Position *before* the update is applied
        # self.pos = 2*current_pos - self.__prev_pos + self.acc*time_passed**2
        # self.__prev_pos = current_pos
        # self.vel += self.acc * time_passed

        self.pos = self.pos + time_passed * self.vel + (1/6)*(time_passed**2)*(
            4*self.__prev_acc - self.__prev_prev_acc)
        self.vel = self.vel + (1/12)*time_passed*(
            5*self.acc + 8*self.__prev_acc - self.__prev_prev_acc)
        


class PhysicalParticle(Particle):
    """
    Particle with mass (negative mass allowed).
    Has a gravity field defines around it, that can locally be inferred.
    """

    def __init__(self,
                 pos: np.ndarray,
                 vel: np.ndarray,
                 acc: np.ndarray,
                 mass: np.ndarray,
                 attraction_function: callable):
        super().__init__(pos, vel, acc)
        self._attraction_function = attraction_function
        self.__forces = set(np.zeros_like(acc))

    # @property
    # def forces(self) -> Set[np.ndarray]:
    #     return self.__forces

    # @forces.setter
    # def forces

    def update(self, time_passed: float, 
                forces: Optional[np.ndarray]=None):
        """
        Update movement of particle according to Newtonian mechanics
        for a period of [time_passed]. Can take a set of forces that influence
        the current acceleration of the particle.
        """
        if forces is not None:
            # Newton's second law
            self.acc = np.sum(forces, axis=0) / self.mass
        super.update(time_passed)
        

    def compute_attraction_force_to(
            self, other: PhysicalParticle) -> np.ndarray:
        """
        Computes the Newtonian gravity force vector induced
        by this particle to the
        [other] paricle at the position of the other particle.
        Note that no gravity constant is included in this computed force.
        """
        difference_vector = (other.pos - self.pos)
        radius =  np.linalg.norm(difference_vector)
        direction = difference_vector /radius
        
        return direction * self._attraction_function(self, other)
