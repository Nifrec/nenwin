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

Class to hold pieces of a network simulation together and manage i/o
around the simulation.
"""
import enum
from typing import Optional, Iterable
import multiprocessing
from numbers import Number

from nenwin.model import NenwinModel
from nenwin.input_placer import InputPlacer
from nenwin.output_reader import OutputReader


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
        self.__num_remaining_timesteps = 0

    @property
    def is_running(self) -> bool:
        return self.__num_remaining_timesteps > 0

    def run(self,
            step_size: Number,
            max_num_steps: Optional[Number] = float("inf")):
        """
        Repeatedly read commands (stop, new inputs, output request),
        and advance the simulation one step.
        Simulation is stopped (paused, not deleted) when the stop command
        is sent via the associated pipe, or optionally after a maximum number
        of steps.

        Arguments:
        * step_size: how much passed time is simulated during a single timestep.
            Lowever values lead to more accurate simulation, but also require
            more computation time.
        * [Optional] max_num_steps: amount of steps after which the simulation
            is guarranteed to stop. 
            It might stop earlier in case of a stop command.
        """
        self.__num_remaining_timesteps = max_num_steps

        while (self.__num_remaining_timesteps > 0):
            self.__num_remaining_timesteps -= 1
            self.__handle_commands()
            self.__model.make_timestep(step_size)

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
                self.__num_remaining_timesteps = 0
            elif command == UICommands.write_output:
                self.__produce_outputs()
            elif command == UICommands.read_input:
                self.__handle_inputs(message.data)

    def __produce_outputs(self):
        output = self.__output_reader.read_output(self.__model)
        self.__pipe.send(output)

    def __handle_inputs(self, data: Iterable):
        new_marbles = self.__input_placer.marblelize_data(data)
        self.__model.add_marbles(new_marbles)
