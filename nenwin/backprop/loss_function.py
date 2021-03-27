"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
March 2021

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

Differentiable loss function for using a Nenwin model for classification.
"""
from typing import List, Sequence
import torch

from nenwin.marble_eater_node import MarbleEaterNode
from nenwin.node import Node
from nenwin.model import NenwinModel
from nenwin.auxliary import distance


class NenwinLossFunction:
    """
    Loss function as described in the report under
    Training/A first experiment: EaterNodes as output/Classification loss.
    """

    def __init__(self,
                 output_nodes: Sequence[MarbleEaterNode],
                 model: NenwinModel):
        """
        Arguments:
            * output_nodes: ordered sequence of Nodes,
                whose index correspond to the
                classification-class they represent.
        """
        self.__output_nodes = output_nodes
        self.__model = model

    def __call__(self, expected: int) -> torch.Tensor:
        activated_nodes = self.__get_activated_nodes()
        assert len(activated_nodes) <= 1, "Multiple Nodes are giving output"

        output_index = self.__output_nodes.index(activated_nodes[0])
        if output_index == expected:
            return torch.tensor([0.0], requires_grad=True)

    def __get_activated_nodes(self) -> List[MarbleEaterNode]:
        """
        Return all Nodes designated as outputs that have eaten one or
        more Marbles.
        """
        return filter(lambda n: n.num_marbles_eaten >= 1, self.__output_nodes)


def find_closest_marble_to(particle: Node, model: NenwinModel):
    """
    Return the closest Marble to the given particle (Node or Marble) 
    in the model.
    [particle] does not need to be in [model] itself, but the dimensions
    must match.
    Raises an error if no Marbles present (or if [particle] is the only Marble).

    NOTE: particle is typed
    """
    other_marbles = model.marbles.difference([particle])
    if len(other_marbles) == 0:
        raise RuntimeError("Model does not contain any Marble that"
                           " is not target particle")
    
    else:
        return min(other_marbles, key=lambda m: distance(m, particle))