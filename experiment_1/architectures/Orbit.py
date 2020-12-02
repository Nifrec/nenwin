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

Architecture implementing the simulation of a semi-stable orbit.
"""
import numpy as np
from typing import Tuple, Optional, Dict
from experiment_1.stiffness_particle import Node, Marble
from experiment_1.attraction_functions.attraction_functions import NewtonianGravity
from experiment_1.architectures.run_and_visualize import run
from experiment_1.auxliary import generate_stiffness_dict
ZERO = np.array([0, 0])

NODE_STIFFNESS = generate_stiffness_dict(marble_stiffness=0,
                                           node_stiffness=0,
                                           marble_attraction=1,
                                           node_attraction=0)
MARBLE_STIFFNESS = generate_stiffness_dict(marble_stiffness=0,
                                               node_stiffness=0,
                                               marble_attraction=1,
                                               node_attraction=0)

#masses modelled afte respective masses of earth and moon
satellite = Marble(np.array([110, 150]), np.array([2, -2]), ZERO, 1.2, NewtonianGravity(), None, **MARBLE_STIFFNESS)
planet = Node(np.array([80, 110]), ZERO, ZERO, 500, NewtonianGravity(), **NODE_STIFFNESS)
planet2 = Node(np.array([200, 190]), ZERO, ZERO, 800, NewtonianGravity(), **NODE_STIFFNESS)

marbles = [satellite]
nodes = [planet, planet2]
run(marbles, nodes)
