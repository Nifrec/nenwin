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

An experiment to see if an architecture can be made that accelerates a
ḿarble in a straight line.
"""
import numpy as np
from typing import List

from experiment_1.model import NenwinModel
from experiment_1.node import Node, MarbleEaterNode
from experiment_1.marble import Marble
from experiment_1.attraction_functions.attraction_functions import Gratan
from experiment_1.attraction_functions.attraction_functions import NewtonianGravity
from experiment_1.visualization import NenwinVisualization

ZERO = np.array([0, 0])
ATTRACTION_FUNCTION = NewtonianGravity()


def nand_gate(input_1: bool, input_2: bool) -> bool:
    """
    Simulate a NAND gate using a Nenwin architecture.
    Also visualizes the simulation and prints the result.
    """
    marble = Marble(pos=np.array([50, 50]),
                    vel=np.array([10, 0]),
                    acc=ZERO,
                    mass=-1,
                    attraction_function=NewtonianGravity,
                    datum=None)
    nodes = __generate_canon_nodes()
    model = NenwinModel(nodes, 0.001, [marble])
    visualization = NenwinVisualization((500, 500), model, 2)
    visualization.run(10)
    output = model._produce_outputs()


def __generate_canon_nodes() -> List[Node]:
    positions = [
        [110, 10], [110, 90],
        [90, 20], [90, 80],
        [70, 30], [70, 70],
        [50, 40], [50, 60],
    ]
    # Exponential masses: (10^n)
    masses = [
        10000, 10000,
        1000, 1000,
        100, 100,
        10, 10,
    ]
    # Quadratic masses (x², starting at x=10):
    # masses = [
    #     100000000, 100000000,
    #     10000, 10000,
    #     100, 100,
    #     10, 10,
    #     1000
    # ]
    nodes = []
    for pos, mass in zip(positions, reversed(masses)):
        new_node = Node(pos=np.array(pos),
                        vel=ZERO,
                        acc=ZERO,
                        mass=mass,
                        attraction_function=ATTRACTION_FUNCTION,
                        stiffness=1)
        nodes.append(new_node)
    return nodes


if __name__ == "__main__":
    nand_gate(True, False)
