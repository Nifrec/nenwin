"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Multi-purpose auxiliary functions.
"""
import numpy as np
from experiment_1.particle import Particle

def distance(p1: Particle, p2: Particle) -> float:
    difference = p1.pos - p2.pos
    return np.linalg.norm(difference)