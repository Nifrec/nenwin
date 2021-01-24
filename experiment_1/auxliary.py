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
import numpy as np
from typing import Dict
import torch

from experiment_1.particle import Particle


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
