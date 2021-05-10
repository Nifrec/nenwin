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

Multi-purpose auxiliary functions.
"""
from typing import Dict, List, Sequence
import torch

from nenwin.particle import Particle
from nenwin.all_particles import MarbleEaterNode, Node
from nenwin.attraction_functions.attraction_functions import AttractionFunction
def distance(p1: Particle, p2: Particle) -> float:
    difference = p1.pos - p2.pos
    return torch.norm(difference)


def generate_stiffness_dict(marble_stiffness: float,
                            node_stiffness: float,
                            marble_attraction: float,
                            node_attraction: float) -> Dict[str, float]:
    return {
        "marble_stiffness": marble_stiffness,
        "node_stiffness": node_stiffness,
        "marble_attraction": marble_attraction,
        "node_attraction": node_attraction
    }


def generate_node_dict(pos: torch.Tensor,
                       vel: torch.Tensor,
                       acc: torch.Tensor,
                       mass: float,
                       attraction_function: callable,
                       marble_stiffness: float,
                       node_stiffness: float,
                       marble_attraction: float,
                       node_attraction: float) -> Dict[str, float]:
    output = {
        "pos": pos,
        "vel": vel,
        "acc": acc,
        "mass": mass,
        "attraction_function": attraction_function
    }
    output.update(generate_stiffness_dict(marble_stiffness,
                                          node_stiffness,
                                          marble_attraction,
                                          node_attraction))
    return output

def gen_nodes(attract_funct: AttractionFunction,
              mass: float,
              positions: Sequence[Sequence[float]]) -> List[Node]:
    """
    Generate a list of Nodes, at the given positions.
    Each Node will use the given attraction_function,
    and will have the given mass.
    Velocity and acceleration will be initialized with a zero vector.
    """
    nodes = []
    num_dims = len(positions[0])
    zero = torch.zeros(num_dims, dtype=torch.float)
    for node_pos in positions():
        node_pos = torch.tensor(node_pos, dtype=torch.float)
        node = Node(pos=node_pos, vel=zero, acc=zero, 
                    mass=mass,
                    attraction_function=attract_funct, 
                    marble_stiffness = 1,
                    node_stiffness = 1,
                    marble_attraction = 1,
                    node_attraction = 0)
        nodes.append(node)
    return nodes

def gen_eater_nodes(attract_funct: AttractionFunction,
              mass: float,
              radius:float,
              positions: Sequence[Sequence[float]]) -> List[MarbleEaterNode]:
    """
    Generate a list of MarbleEaterNodes, at the given positions.
    Each Node will use the given attraction_function,
    and will have the given mass and radius.
    Velocity and acceleration will be initialized with a zero vector.
    """
    eaters = []
    num_dims = len(positions[0])
    zero = torch.zeros(num_dims, dtype=torch.float)
    for eater_pos in positions:
        eater_pos = torch.tensor(eater_pos, dtype=torch.float)
        eater = MarbleEaterNode(pos=eater_pos, vel=zero, acc=zero, 
                    mass=mass,
                    attraction_function=attract_funct, 
                    marble_stiffness = 1,
                    node_stiffness = 1,
                    marble_attraction = 1,
                    node_attraction = 0,
                    radius = radius)
        eaters.append(eater)
    return eaters