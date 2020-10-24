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
import multiprocessing.connection
import enum
from numbers import Number

from experiment_1.particle import PhysicalParticle
from experiment_1.node import Node
from experiment_1.node import MarbleEaterNode
from experiment_1.marble import Marble
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
        self.__eater_nodes: List[MarbleEaterNode] =\
            [node for node in nodes if isinstance(node, MarbleEaterNode)]
        if initial_marbles is not Node:
            self.__marbles = set(initial_marbles)
        else:
            initial_marbles = set()
        self.__all_particles = self.__nodes.union(self.__marbles)
        self.__step_size = step_size
        self.__pipe_end, self.__other_pipe_end = multiprocessing.Pipe(True)

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
    def pipe(self) -> multiprocessing.connection.Connection:
        """
        Return reference to the UI's side of the input/output pipe,
        i.e. a multiprocessing.connection.Connection instance.
        """
        return self.__other_pipe_end

    def run(self, max_num_steps: Number = float("inf")):
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

            self.__handle_commands()

            for particle in self.__all_particles:
                net_force = self.__compute_net_force_for(particle)
                particle.update_acceleration(net_force)

            for particle in self.__all_particles:
                particle.update_movement(self.__step_size)

            for marble in self.__marbles:
                self.__feed_marble_if_close_to_any_eater(marble)


    def __compute_net_force_for(self, particle: PhysicalParticle) -> np.ndarray:
        """
        Compute net force for [particle] as excerted on it by 
        all other particles in the simulation.
        The result is a 2D matrix with a single row.
        """
        forces = np.zeros_like(particle.acc)
        other_particles = self.__all_particles.difference((particle,))
        for other_particle in other_particles:
            forces += other_particle.compute_attraction_force_to(particle)
        return np.array([forces])

    def __handle_commands(self):
        """
        Reads command given through the pipe and
        executes it. 
        Does nothing if no command exists in the queue.
        """
        if self.__pipe_end.poll():
            message = self.__pipe_end.recv()
            assert isinstance(message, UIMessage)

            command = message.command
            if command == UICommands.stop:
                assert False, "TODO"
            elif command == UICommands.write_output:
                self.__produce_outputs()
            elif command == UICommands.read_input:
                self.__handle_inputs(message.data)

    def __feed_marble_if_close_to_any_eater(self, marble: Marble):
        warnings.warn("Method not tested yet")
        for eater in self.__eater_nodes:
            if aux.distance(marble, eater) <= eater.radius:
                eater.eat(marble)
                self.__marbles.remove(marble)
                break

    def __handle_inputs(self, inputs):
        assert False, "TODO"
        pass

    def __produce_outputs(self):
        assert False, "TODO"
        pass
