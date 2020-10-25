"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

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

Author: Lulof Pirée
October 2020

Simulation of NAND-gates using Nenwin architectures.
"""
import numpy as np

from experiment_1.model import NumMarblesEatenAsOutputModel
from experiment_1.node import Node, MarbleEaterNode
from experiment_1.marble import Marble
from experiment_1.attraction_functions.attraction_functions import Gratan
from experiment_1.attraction_functions.attraction_functions import NewtonianGravity
from experiment_1.visualization import NenwinVisualization

BOOL_TO_MASS = {True: 10, False: -10}


def nand_gate(input_1: bool, input_2: bool) -> bool:
    """
    Simulate a NAND gate using a Nenwin architecture.
    Also visualizes the simulation and prints the result.
    """
    attraction_function = NewtonianGravity()
    zero = np.array([0, 0])
    eater = MarbleEaterNode(pos=np.array([10, 20]),
                            vel=zero,
                            acc=zero,
                            mass=10,
                            stiffness=1,
                            attraction_function=attraction_function,
                            radius=5)

    node = Node(pos=np.array([30, 20]),
                vel=zero,
                acc=zero,
                mass=-10,
                stiffness=1,
                attraction_function=attraction_function)

    marble1 = Marble(np.array([20, 10]),
                     vel=zero, acc=zero, mass=BOOL_TO_MASS[input_1],
                     attraction_function=attraction_function,
                     datum=input_1)

    marble2 = Marble(np.array([20, 30]),
                     vel=zero, acc=zero, mass=BOOL_TO_MASS[input_2],
                     attraction_function=attraction_function,
                     datum=input_2)

    model = NumMarblesEatenAsOutputModel([eater, node],
                                         0.01,
                                         [marble1, marble2])
    visualization = NenwinVisualization((500, 500), model)
    visualization.run(10)


if __name__ == "__main__":
    nand_gate(True, False)
