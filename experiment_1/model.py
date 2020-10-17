"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Class to hold pieces of network together, perform timesteps and manage i/o
around the simulation.
"""
import numpy as np
from typing import Iterable
import multiprocessing

from experiment_1.particle import PhysicalParticle
from experiment_1.node import Node

class NenwinModel():
    """
    Class representing the simulation: keeps track of the nodes, marbles,
    input, output and advances timesteps.
    """
    
    def __init__(self, nodes: Iterable[Node], step_size: float):
        """
        Set up the model, given a number of Node instances,
        and a step-size for the simulation.
        """
        self.__nodes = nodes
        self.__step_size = step_size
        self.__queue = multiprocessing.Queue()

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

    def run(self):
        pass