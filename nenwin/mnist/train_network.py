
"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Teun Schilperoort, Lulof Pirée
April 2021

Copyright (C) 2021 Lulof Pirée, Teun Schilperoort

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


Auxiliary function for training a NenwinTrainer instance tailored
to train on the MNIST dataset.
"""

from nenwin.constants import MNIST_CHECKPOINT_DIR
import torch
import torch.nn
from typing import Iterable

from nenwin.all_particles import MarbleEaterNode, Marble, Node, MarbleEmitterNode
from nenwin.model import NenwinModel
from nenwin.input_placer import InputPlacer
from nenwin.backprop.filename_gen import FilenameGenerator
from nenwin.backprop.trainer import NenwinTrainer
from nenwin.backprop.training_stats import TrainingStats
from nenwin.backprop.loss_function import NenwinLossFunction
from nenwin.mnist.load_dataset import load_mnist_dataset
from nenwin.mnist.load_dataset import MNISTDataset


def create_trainer(model: NenwinModel,
                   input_placer: InputPlacer,
                   output_nodes: Iterable[MarbleEaterNode],
                   dataset: MNISTDataset,
                   loss_pos_weight: float,
                   loss_vel_weight: float,
                   ):

    loss_funct = NenwinLossFunction(output_nodes, model, loss_vel_weight,
                                    loss_pos_weight)
    optim = torch.optim.Adam(model.parameters())
    filename_gen = FilenameGenerator(MNIST_CHECKPOINT_DIR, "MNIST_", ".txt")
    trainer = NenwinTrainer(model, loss_funct, optim, filename_gen,
                            input_placer, dataset)
    return trainer
