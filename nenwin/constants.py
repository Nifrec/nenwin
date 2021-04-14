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

Collection of constant values.
"""
import torch
import os

import nenwin

# Maximum distance at which an EmitterNode can spawn a Marble
# (measured from the border of the radius, towards the pos of the EmitterNode)
MAX_EMITTER_SPAWN_DIST = 1e-3

# Device on which Tensors are stored. Either GPU or CPU
DEVICE = torch.device('cpu')


# Directory where the nenwin repository is located in the OS´ filesystem.
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(nenwin.__file__))

MNIST_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "nenwin", "mnist", "dataset")

MNIST_CACHE_FILE = os.path.join(PROJECT_ROOT_DIR, "nenwin", "mnist", "dataset",
                                "cached_dataset.pickle")
