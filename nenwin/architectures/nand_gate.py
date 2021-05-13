"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Copyright (C) 2020 Lulof Pirée, 

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

Author: Lulof Pirée
October 2020

Simulation of NAND-gates using Nenwin architectures.
"""
import numpy as np
from typing import Tuple

from nenwin.simulation import Simulation
from nenwin.model import NenwinModel
from nenwin.node import Node, Marble
from nenwin.marble_eater_node import MarbleEaterNode
from nenwin.attraction_functions.attraction_functions import NewtonianGravity
from nenwin.visualization import NenwinVisualization
from nenwin.output_reader import NumMarblesOutputReader

BOOL_TO_MASS = {True: 100, False: -100}
ZERO = np.array([0, 0])
ATTRACTION_FUNCTION = NewtonianGravity()
STIFFNESS_AND_ATTRACTION = {
    "marble_stiffness": 1,
    "node_stiffness": 0,
    "marble_attraction": 1,
    "node_attraction": 0
}
STEP_SIZE = 0.001

class MockPipe:

    def poll(self):
        return None

def nand_gate(input_1: bool, input_2: bool) -> bool:
    """
    Simulate a NAND gate using a Nenwin architecture.
    Also visualizes the simulation and prints the result.
    """
    print(f"NAND({input_1}, {input_2})")
    print("Press ESC after the simulation ended to print NAND gate output")
    marble1, marble2 = __generate_nand_marbles(input_1, input_2)
    eater, wastebin_eater = __generate_nand_nodes()
    model = NenwinModel([eater, wastebin_eater], [marble1, marble2],)
    simulation = Simulation(model, None, None, MockPipe())
    visualization = NenwinVisualization((500, 500), simulation, model, 10)
    visualization.run(10, STEP_SIZE)
    output = NumMarblesOutputReader().read_output(model)

    print("NAND output:")
    if output[0] == 2:
        print(False)
    else:
        print(True)


def __generate_nand_marbles(input_1: bool, input_2: bool) -> Tuple[Marble]:
    marble1 = Marble(np.array([20, 10]),
                     vel=ZERO, acc=ZERO, mass=BOOL_TO_MASS[input_1],
                     attraction_function=ATTRACTION_FUNCTION,
                     datum=input_1,
                     **STIFFNESS_AND_ATTRACTION)

    marble2 = Marble(np.array([20, 30]),
                     vel=ZERO, acc=ZERO, mass=BOOL_TO_MASS[input_2],
                     attraction_function=ATTRACTION_FUNCTION,
                     datum=input_2,
                     **STIFFNESS_AND_ATTRACTION)
    return marble1, marble2


def __generate_nand_nodes() -> Tuple[MarbleEaterNode]:
    eater = MarbleEaterNode(pos=np.array([5, 20]),
                            vel=ZERO,
                            acc=ZERO,
                            mass=10,
                            attraction_function=ATTRACTION_FUNCTION,
                            radius=5,
                            **STIFFNESS_AND_ATTRACTION)

    wastebin_eater = MarbleEaterNode(pos=np.array([35, 20]),
                                     vel=ZERO,
                                     acc=ZERO,
                                     mass=-10,
                                     attraction_function=ATTRACTION_FUNCTION,
                                     radius=5,
                                     **STIFFNESS_AND_ATTRACTION)
    return eater, wastebin_eater


if __name__ == "__main__":
    nand_gate(True, False)
