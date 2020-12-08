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

Unit-tests for Simulation of simulation.py.
"""
import unittest
import numpy as np
import multiprocessing

from experiment_1.input_placer import InputPlacer
from experiment_1.output_reader import OutputReader
from experiment_1.simulation import Simulation, UICommands, UIMessage
from experiment_1.model import NenwinModel
from experiment_1.node import Marble
from test_aux import ZERO

# Just an arbitrary non-trivial object
# ('None' would also be returned when faultly no return value is specified)
MOCK_KEYWORD = "test"
MOCK_MARBLE = Marble(ZERO, ZERO, ZERO, 0, None, None)


class SimulationTestCase(unittest.TestCase):

    def setUp(self):
        self.model = MockNenwinModel()
        self.input_placer = MockInputPlacer()
        self.output_reader = MockOutputReader()
        self.pipe, self.__other_pipe_end = multiprocessing.Pipe(duplex=True)
        self.simulation = Simulation(self.model,
                                     self.input_placer,
                                     self.output_reader,
                                     self.__other_pipe_end)

    def test_handle_commands_stop(self):
        """
        Base case: test if stop command executed.
        """
        self.send_command_and_let_process(UICommands.stop)
        self.assertFalse(self.simulation.is_running)

    def test_write_output_command(self):
        """
        Base case: test if write_output command executed.
        """
        self.send_command_and_let_process(UICommands.write_output)
        self.assertEqual(self.output_reader.invocation_count, 1)

        output = self.pipe.recv()
        self.assertEqual(output, MOCK_KEYWORD)

    def test_read_input_command(self):
        """
        Base case: test if write_output command executed.
        """
        self.send_command_and_let_process(UICommands.read_input)
        self.assertEqual(self.input_placer.invocation_count, 1)

        self.assertTrue(MOCK_MARBLE in self.model.marbles)

    def test_commands_executed_multiple_times(self):
        """
        Base case: twice same command.
        """
        self.send_command_and_let_process(UICommands.read_input)
        self.send_command_and_let_process(UICommands.read_input)
        self.assertEqual(self.input_placer.invocation_count, 2)

    def test_run_1(self):
        """
        Base case: test for termination.
        """
        self.simulation.run(step_size=1, max_num_steps=1)
        self.assertFalse(self.simulation.is_running)

    def test_run_2(self):
        """
        Base case: test if model's step function was called max_num_steps times.
        """
        num_steps = 10
        self.simulation.run(step_size=1, max_num_steps=num_steps)
        self.assertEqual(self.model.invocation_count, num_steps)

    def test_run_3(self):
        """
        Base case: stop after one step if stop command on pipe.
        """
        max_num_steps = 10
        message = UIMessage(UICommands.stop)
        self.pipe.send(message)
        self.simulation.run(step_size=1, max_num_steps=max_num_steps)
        self.assertEqual(self.model.invocation_count, 1)

    def send_command_and_let_process(self, command: UICommands):
        """
        Send a single command to self.simulation,
        and call self.simulation._Simulation__hangle_commands() once.
        """
        message = UIMessage(command)
        self.pipe.send(message)
        self.simulation._Simulation__handle_commands()


class CheckInvokedMocker():
    def __init__(self):
        self.invocation_count = 0

    def count_invocation(self):
        self.invocation_count += 1


class MockInputPlacer(CheckInvokedMocker, InputPlacer):
    def __init__(self):
        CheckInvokedMocker.__init__(self)
        InputPlacer.__init__(self, ZERO, ZERO)

    def marblize_data(self, input_data):
        self.count_invocation()
        return [MOCK_MARBLE]


class MockOutputReader(CheckInvokedMocker, OutputReader):
    def __init__(self):
        CheckInvokedMocker.__init__(self)
        OutputReader.__init__(self)

    def read_output(self, model):
        self.count_invocation()
        return MOCK_KEYWORD


class MockNenwinModel(CheckInvokedMocker, NenwinModel):
    def __init__(self):
        CheckInvokedMocker.__init__(self)
        NenwinModel.__init__(self, [], [])

    def make_timestep(self, time_passed):
        self.count_invocation()


if __name__ == '__main__':
    unittest.main()
