"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Class to hold pieces of network together, perform timesteps and manage i/o
around the simulation.
"""
import numpy as np
from typing import Set, Iterable, Optional
import multiprocessing
from numbers import Number

from experiment_1.particle import PhysicalParticle
from experiment_1.node import Node
from experiment_1.marble import Marble


class NenwinModel():
    """
    Class representing the simulation: keeps track of the nodes, marbles,
    input, output and advances timesteps.
    """

    def __init__(self,
                 nodes: Iterable[Node],
                 step_size: float,
                 initial_marbles: Optional[Set[Marble]] = None):
        """
        Set up the model, given a number of Node instances,
        and a step-size for the simulation.
        Optionally some input Marble's can already be added, 
        which will move as soon as NenwinModel.run() is called.
        """
        self.__nodes = set(nodes)
        self.__marbles = set()
        self.__step_size = step_size
        self.__queue = multiprocessing.Queue()

    @property
    def nodes(self) -> Set[Node]:
        """
        Get the set of all currently simulated nodes, 
        each in their current state.
        """
        return self.__nodes.copy()

    @property
    def marbles(self) -> Set[Marble]:
        """
        Get the set of all currently simulated marbles, 
        each in their current state.
        """
        return self.__marbles.copy()

    @property
    def queue(self):
        """
        Return reference to input/output multiprocessing.Queue
        """
        return self.__queue

    def __handle_inputs(self):
        pass

    def __produce_outputs(self):
        pass

    def run(self, max_num_steps:Number=float("inf")):
        """
        Start simulation and imput processing until stop signal is received.
        While running, will accept inputs, and produce outputs when requested.

        By default runs indefinitely until stop signal is received.
        An optional max amount of timesteps can be given
        (convenient for testing).
        """
        num_remaining_steps = max_num_steps

        while num_remaining_steps > 0:
            num_remaining_steps -= 1

