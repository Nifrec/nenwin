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

Demonstration of the visualization of a Nenwin model.
"""
import numpy as np

from experiment_1.visualization import NenwinVisualization
from experiment_1.model import NenwinModel
from experiment_1.node import Marble
from experiment_1.node import Node
from experiment_1.simulation import Simulation
from experiment_1.attraction_functions.attraction_functions import Gratan, NewtonianGravity
from test_aux import MockPipe

def visualization_demo():
    attract_funct = NewtonianGravity()
    marbles = set()
    marble_1 = Marble(np.array([460, 460]),
        np.array([0, 0]),
        np.array([0, 0]),
        mass=20,
        attraction_function=attract_funct,
        datum=None,
        marble_stiffness=1,
        node_stiffness=0, 
        marble_attraction=0, 
        node_attraction=1)
    marble_2 = Marble(np.array([550, 550]),
        np.array([10, 10]),
        np.array([0, 0]),
        mass=40,
        attraction_function=attract_funct,
        datum=None,
        marble_stiffness=1,
        node_stiffness=0, 
        marble_attraction=0, 
        node_attraction=1)
    marble_3 = Marble(np.array([450, 500]),
        np.array([2, -2]),
        np.array([0, 0]),
        mass=40,
        attraction_function=attract_funct,
        datum=None,
        marble_stiffness=1,
        node_stiffness=0, 
        marble_attraction=0, 
        node_attraction=1)
    marbles = set((marble_1, marble_2, marble_3))

    node_1 = Node(np.array([500, 500]),
        np.array([0, 0]),
        np.array([0, 0]),
        mass=200,
        attraction_function=attract_funct, 
        marble_stiffness=1,
        node_stiffness=1, 
        marble_attraction=1, 
        node_attraction=1)

    node_2 = Node(np.array([400, 400]),
        np.array([0, 0]),
        np.array([0, 0]),
        mass=200,
        attraction_function=attract_funct,
        marble_stiffness=0.7,
        node_stiffness=1, 
        marble_attraction=1, 
        node_attraction=1)
    nodes = set((node_1, node_2))


    model = NenwinModel(nodes, marbles)
    simulation = Simulation(model, None, None, MockPipe())
    visualization = NenwinVisualization((1500, 1000), simulation, model)
    visualization.run(10, 0.01)

if __name__ == "__main__":
    visualization_demo()