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
from typing import Callable, List, Sequence, Set
import torch

from nenwin.marble_eater_node import MarbleEaterNode
from nenwin.node import Marble, Node
from nenwin.model import NenwinModel
from nenwin.auxliary import distance





def find_closest_marble_to(particle: Node, model: NenwinModel):
    """
    Return the closest Marble to the given particle (Node or Marble) 
    in the model.
    [particle] does not need to be in [model] itself, but the dimensions
    must match.
    Raises an error if no Marbles present (or if [particle] is the only Marble).

    NOTE: particle is typed as a Node, but Marbles are subclasses of Node
    in this implementation.
    """
    other_marbles = __get_other_marbles(model, particle)
    return min(other_marbles, key=lambda m: distance(m, particle))


def find_most_promising_marble_to(particle: Node,
                                  model: NenwinModel,
                                  pos_weight: float,
                                  vel_weight: float):
    """
    Find the Marble m where 
        ||particle.pos - (pos_weight * m.pos + vel_weight * m.vel)||²
    is smallest, where m !- particle.

    [particle] does not need to be in [model] itself, but the dimensions
    must match.
    Raises an error if no Marbles present (or if [particle] is the only Marble).

    NOTE: particle is typed as a Node, but Marbles are subclasses of Node
    in this implementation.
    """
    other_marbles = __get_other_marbles(model, particle)

    def key(m): 
        return velocity_weighted_distance(particle, m, pos_weight, vel_weight)

    return min(other_marbles, key=key)


def velocity_weighted_distance(stationary: Node,
                               moving: Node,
                               pos_weight: float,
                               vel_weight: float):
    """
    Return:
    ||stationary.pos - (pos_weight * moving.pos + vel_weight * moving.vel)||²
    """
    output = stationary.pos - (pos_weight*moving.pos + vel_weight*moving.vel)
    return torch.square(torch.norm(output))


def __get_other_marbles(model: NenwinModel, target: Node) -> Set[Marble]:

    other_marbles = model.marbles.difference([target])
    if len(other_marbles) == 0:
        raise RuntimeError("Model does not contain any Marble that"
                           " is not target particle")
    return other_marbles

class NenwinLossFunction:
    """
    Loss function as described in the report under
    Training/A first experiment: EaterNodes as output/Classification loss.
    """

    def __init__(self,
                 output_nodes: Sequence[MarbleEaterNode],
                 model: NenwinModel,
                 vel_weight: float,
                 pos_weight: float = 1):
        """
        Arguments:
            * output_nodes: ordered sequence of Nodes,
                whose index correspond to the
                classification-class they represent.
            * model: NenwinModel containing the output_nodes,
                used to compute the loss.
        """
        if not all(node in model.nodes for node in output_nodes):
            raise RuntimeError("Not all output_nodes are in model")
        self.__output_nodes = output_nodes
        self.__model = model

    def __call__(self, expected: int) -> torch.Tensor:
        activated_nodes = self.__get_activated_nodes()
        assert len(activated_nodes) <= 1, "Multiple Nodes are giving output"

        output_index = self.__output_nodes.index(activated_nodes[0])
        if output_index == expected:  # Correct prediction
            return torch.tensor([0.0], requires_grad=True)
        else:  # "Blame" it on closest Marble.
            ...
    def __get_activated_nodes(self) -> List[MarbleEaterNode]:
        """
        Return all Nodes designated as outputs that have eaten one or
        more Marbles.
        """
        return filter(lambda n: n.num_marbles_eaten >= 1, self.__output_nodes)