"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Class to hold pieces of a network simulation together and manage i/o
around the simulation.
"""
import enum
from typing import Optional, Iterable
import multiprocessing
from numbers import Number

from experiment_1.model import NenwinModel
from experiment_1.input_placer import InputPlacer
from experiment_1.output_reader import OutputReader

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

class Simulation():
    """
    Class to hold pieces of a network simulation together
    and manage i/o around the simulation.
    """

    def __init__(self,
                 model: NenwinModel,
                 input_placer: InputPlacer,
                 output_reader: OutputReader,
                 pipe_end: multiprocessing.connection.Connection):
        self.__model = model
        self.__input_placer = input_placer
        self.__output_reader = output_reader
        self.__pipe = pipe_end
        self.__is_running = False

    @property
    def is_running(self) -> bool:
        return self.__is_running

    def run(max_num_steps: Number = float("inf")):
        self.__is_running = True
        pass

    def __handle_commands(self):
        """
        Reads command given through the pipe and
        executes it. 
        Does nothing if no command exists in the queue.
        """
        if self.__pipe.poll():
            message = self.__pipe.recv()
            assert isinstance(message, UIMessage)

            command = message.command
            if command == UICommands.stop:
                self.__is_running = False
            elif command == UICommands.write_output:
                self.__produce_outputs()
            elif command == UICommands.read_input:
                self.__handle_inputs(message.data)

    def __produce_outputs(self):
        output = self.__output_reader.read_output(self.__model)
        self.__pipe.send(output)

    def __handle_inputs(self, data: Iterable):
        new_marbles = self.__input_placer.marblize_data(data)
        self.__model.add_marbles(new_marbles)
    # def run(self, max_num_steps: Number = float("inf")):
    #     """
    #     Start simulation and imput processing until stop signal is received.
    #     While running, will accept inputs, and produce outputs when requested.

    #     By default runs indefinitely until stop signal is received.
    #     An optional max amount of timesteps can be given
    #     (convenient for testing).
    #     """
    #     num_remaining_steps = max_num_steps

    #     while num_remaining_steps > 0:
    #         num_remaining_steps -= 1

    #         self.__handle_commands()

    #         