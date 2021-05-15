"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
May 2021

Copyright (C) 2021 Lulof Pirée, 

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

--------------------------------------------------------------------------------

Collection of functions that generate a Nenwin architecture
to train on the banknote dataset. 
The architectures are simply called A, B and C.
"""
import torch
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Iterable
from numbers import Number
import enum

from nenwin.constants import BANKNOTE_CHECKPOINT_DIR
from nenwin.all_particles import Marble, Node, MarbleEmitterNode, MarbleEaterNode
from nenwin.model import NenwinModel
from nenwin.input_placer import InputPlacer
from nenwin.grid_input_placer import VelInputPlacer
from nenwin.attraction_functions.attraction_functions import NewtonianGravity, AttractionFunction
from nenwin.backprop.filename_gen import FilenameGenerator
from nenwin.backprop.trainer import NenwinTrainer
from nenwin.backprop.training_stats import TrainingStats
from nenwin.backprop.loss_function import NenwinLossFunction
from nenwin.banknote_dataset.load_dataset import load_banknote_dataset, BanknoteDataset
from nenwin.creation_functions import gen_nodes, gen_eater_nodes
from nenwin.plot_model import plot_model


class ARCHITECTURES(enum.Enum):
    A = "A"
    B = "B"
    C = "C"


def gen_architecture(which: ARCHITECTURES
                     ) -> Tuple[NenwinModel, VelInputPlacer, Tuple[Node]]:
    """
    Generate either architecture A, B or C.
    All architectures have the same input region,
    and all have two output MarbleEaterNodes,
    but the number & location of the other nodes vary.
    Arguments:
        * which: indicates which of the three architectures should
            be generated.

    Returns:
        * Model holding the generated architecture.
        * VelInputPlacer adapted to the architecture.
        * Tuple of the two MarbleEaterNodes of the architecture.
    """

    if which == ARCHITECTURES.A:
        return gen_architecture_a()
    if which == ARCHITECTURES.B:
        raise NotImplementedError()
    elif which == ARCHITECTURES.C:
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown architecture label '{which}'")

def gen_architecture_a() -> Tuple[NenwinModel, VelInputPlacer, Tuple[Node]]:
    """
    Generate the following architecture:
        * The input region is at (-2.5, -1) and has size (5, 2) 
            (So it has vertices {(-2.5, -1), (-2.5, 1), (2.5, -1), (2.5, 1)})
        * There are two MarbleEaterNodes, at (-10, 0) and (10, 0)
        * There are four normal Nodes, at (0, -5), (-5, 0), (5, 0) and (0, 5).

    Returns:
        * Model holding the architecture descibed above
        * VelInputPlacer with the input region as described above.
        * Tuple of the two MarbleEaterNodes
    """

    eater_positions = [(-10, 0), (10, 0)]
    node_positions = [(0, -5), (-5, 0), (5, 0), (0, 5)]
    input_region_pos = np.array((-2.5, -1))
    input_region_size = np.array((5, 2))
    mass = 1
    radius = 0.5

    attraction_function = NewtonianGravity()

    nodes = gen_nodes(attraction_function, mass, node_positions)
    eater_nodes = gen_eater_nodes(attraction_function, mass,
                                  radius, eater_positions)
    model = NenwinModel(nodes+eater_nodes)
    input_placer = VelInputPlacer(input_region_pos, input_region_size)

    return model, input_placer, eater_nodes
