"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Demonstration of the visualization of a Nenwin model.
"""
import numpy as np

from experiment_1.visualization import NenwinVisualization
from experiment_1.model import NenwinModel
from experiment_1.marble import Marble
from experiment_1.node import Node
from experiment_1.attraction_functions.attraction_functions import Gratan, NewtonianGravity

def visualization_demo():
    attract_funct = NewtonianGravity()
    marbles = set()
    marble_1 = Marble(np.array([460, 460]),
        np.array([0, 0]),
        np.array([0, 0]),
        mass=20,
        attraction_function=attract_funct,
        datum=None)
    marble_2 = Marble(np.array([550, 550]),
        np.array([10, 10]),
        np.array([0, 0]),
        mass=40,
        attraction_function=attract_funct,
        datum=None)
    marble_3 = Marble(np.array([450, 500]),
        np.array([2, -2]),
        np.array([0, 0]),
        mass=40,
        attraction_function=attract_funct,
        datum=None)
    marbles = set((marble_1, marble_2, marble_3))

    node_1 = Node(np.array([500, 500]),
        np.array([0, 0]),
        np.array([0, 0]),
        mass=200,
        attraction_function=attract_funct,
        stiffness=1)

    node_2 = Node(np.array([400, 400]),
        np.array([0, 0]),
        np.array([0, 0]),
        mass=200,
        attraction_function=attract_funct,
        stiffness=0.7)
    nodes = set((node_1, node_2))


    model = NenwinModel(nodes, 0.01, marbles)
    visualization = NenwinVisualization((1500, 1000), model)
    visualization.run(10)

if __name__ == "__main__":
    visualization_demo()