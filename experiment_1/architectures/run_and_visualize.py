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

Auxiliary file that performs a standard setup for
only-at-start-input experiments:
only the nodes, marbles and attraction function need to be specified.
"""

import numpy as np
from typing import Tuple, Iterable, Optional, Any
import math

from experiment_1.simulation import Simulation
from experiment_1.model import NenwinModel
from experiment_1.node import Node, Marble
from experiment_1.marble_eater_node import MarbleEaterNode
from experiment_1.attraction_functions.attraction_functions import AttractionFunction
from experiment_1.visualization import NenwinVisualization
from experiment_1.output_reader import NumMarblesOutputReader, OutputReader

ZERO = np.array([0, 0])
STEP_SIZE = 0.001
SIMULATION_STEPS_PER_FRAME = 10
RESOLUTION = (1000, 1000)
SCALE_FACTOR = 10


class MockPipe:

    def poll(self):
        return None


def run(marbles: Iterable[Marble],
        nodes: Iterable[Node],
        output_reader: Optional[OutputReader] = NumMarblesOutputReader()) -> Any:
    """
    Simulate and visualize a only-at-start-input experiment:
    only the nodes, marbles and attraction function need to be specified.
    """
    model = NenwinModel(nodes=nodes, initial_marbles=marbles)
    simulation = Simulation(model, None, None, MockPipe())
    scale_factor = compute_scale_factor(set(marbles).union(set(nodes)))
    visualization = NenwinVisualization(
        RESOLUTION, simulation, model, scale_factor)
    visualization.run(SIMULATION_STEPS_PER_FRAME, STEP_SIZE)
    return output_reader.read_output(model)

def compute_scale_factor(particles):
    max_x = max([particle.pos[0] for particle in particles])
    max_y = max([particle.pos[1] for particle in particles])
    return math.floor(min(RESOLUTION[0]/max_x, RESOLUTION[1]/max_y))//2