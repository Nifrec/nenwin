"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
March 2021

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

Differentiable loss function for using a Nenwin model for classification.
"""
from typing import Callable, List, Optional, Sequence, Set, Tuple
import torch
from enum import Enum
import warnings

from nenwin.marble_eater_node import MarbleEaterNode
from nenwin.node import Marble, Node
from nenwin.model import NenwinModel
from nenwin.auxliary import distance


class LossCases(Enum):
    correct_prediction = 0
    no_prediction = 1
    wrong_prediction = 2


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
                                  vel_weight: float) -> Marble:
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

def find_min_weighted_distance_to(particle: Node,
                                  model: NenwinModel,
                                  pos_weight: float,
                                  vel_weight: float) -> Marble:
    """
    Same as find_most_promising_marble_to(), but returns
    the weighted distance instead of the Marble.
    """
    other_marbles = __get_other_marbles(model, particle)

    def key(m):
        return velocity_weighted_distance(particle, m, pos_weight, vel_weight)

    return min(map(key, other_marbles))


def velocity_weighted_distance(stationary: Node,
                               moving: Node,
                               pos_weight: float,
                               vel_weight: float) -> torch.Tensor:
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
                 pos_weight: Optional[float] = 1):
        """
        Arguments:
            * output_nodes: ordered sequence of Nodes,
                whose index correspond to the
                classification-class they represent.
            * model: NenwinModel containing the output_nodes,
                used to compute the loss.
            * vel_weight: weight of velocity in computing which
                Marble is to 'blame' for any loss.
                Velocity multiplied with [vel_weight] is added to
                compute the closest marble to the target output Node.
            * pos_weight: idem as vel_weight, but for position. 1 by default.
        """
        if not all(node in model.nodes for node in output_nodes):
            raise RuntimeError("Not all output_nodes are in model")
        self.__output_nodes = output_nodes
        self.__model = model
        self.__vel_weight = vel_weight
        self.__pos_weight = pos_weight

    def __call__(self, expected: int) -> torch.Tensor:

        current_case = self._find_loss_case(expected)

        if current_case == LossCases.correct_prediction:
            return torch.tensor([0.0], requires_grad=True)

        elif current_case == LossCases.no_prediction:
            return self.__compute_loss_case_no_prediction(expected)

        elif current_case == LossCases.wrong_prediction:
            return self.__compute_loss_case_wrong_prediction(expected)

        else:
            raise RuntimeError("Unrecognized case of loss function. "
                               "This can never happen.")

    def __compute_loss_case_no_prediction(self, expected: int):
        target_node = self.__output_nodes[expected]
        blamed_marble = find_most_promising_marble_to(
            self.__output_nodes[expected],
            self.__model,
            self.__pos_weight,
            self.__vel_weight)
        warnings.warn("Inefficiency: vel-weighted-distance is computed "
                      "twice for blamed Marble")
        return velocity_weighted_distance(target_node, blamed_marble,
                                          self.__pos_weight, self.__vel_weight)

    def __compute_loss_case_wrong_prediction(self, expected: int):
        

        activated_node = self.__get_activated_nodes()[0]
        wrong_marble = activated_node.marbles_eaten[0]
        # Negative term for Marble at wrong EaterNode
        loss = -1/velocity_weighted_distance(activated_node, wrong_marble,
                pos_weight=self.__pos_weight, vel_weight=self.__vel_weight)

        # Positive loss term for missing Marble at expected EaterNode.
        try:
            loss += self.__compute_loss_case_no_prediction(expected)
        except RuntimeError:
            # There are no other Marble than the eaten Marble.
            # So *also* blame the eaten Marble for not being at the right place.
            target_node = self.__output_nodes[expected]
            loss += velocity_weighted_distance(target_node, wrong_marble,
                pos_weight=self.__pos_weight, vel_weight=self.__vel_weight)
        
        return loss

    def __get_activated_nodes(self) -> Tuple[MarbleEaterNode]:
        """
        Return all Nodes designated as outputs that have eaten one or
        more Marbles.
        """
        output = filter(lambda n: n.num_marbles_eaten >=
                        1, self.__output_nodes)
        return tuple(output)

    def _find_loss_case(self, expected: int) -> LossCases:
        """
        The loss is a piecewise function.
        This method finds which of the three pieces (cases) currently applies.
        The cases are:
            * No prediction is made, as no Marble has been eaten 
                by any of the output Nodes.
            * The correct prediction has been made 
                (i.e. the correct Node has eaten a Marble)
            * A wrong prediction has been made
                (i.e. a Marble has been eaten but by the wrong Node)
        """
        activated_nodes = self.__get_activated_nodes()
        assert len(activated_nodes) <= 1, "Multiple Nodes are giving output"

        if len(activated_nodes) == 0:
            return LossCases.no_prediction
        else:
            output_index = self.__output_nodes.index(activated_nodes[0])
            if output_index == expected:  # Correct prediction
                return LossCases.correct_prediction
            else:
                return LossCases.wrong_prediction