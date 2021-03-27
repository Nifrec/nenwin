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

A variant of the Node that can 'consume' Marbles if they come close.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import re

from nenwin.node import Marble, Node


class MarbleEaterNode(Node):
    """
    A node variant that consumes Marbles when they become within
    a certain radius of the Eater.
    Keeps track of total count of Marbles eater,
    and the data fields of the Marbles it ate.
    """

    def __init__(self,
                 pos: torch.Tensor,
                 vel: torch.Tensor,
                 acc: torch.Tensor,
                 mass: float,
                 attraction_function: callable,
                 marble_stiffness: float,
                 node_stiffness: float,
                 marble_attraction: float,
                 node_attraction: float,
                 radius: float):
        super().__init__(pos, vel, acc, mass,
                         attraction_function,
                         marble_stiffness,
                         node_stiffness,
                         marble_attraction,
                         node_attraction)

        self.__radius = radius
        self.__num_marbles_eaten: int = 0
        self.__marble_data_eaten = []

    def __repr__(self):
        output = super().__repr__()
        output = "MarbleEater" + output
        output = re.sub(r"\)$", f",{self.radius})", output)
        return output

    @property
    def radius(self) -> float:
        return self.__radius

    @property
    def num_marbles_eaten(self) -> int:
        return self.__num_marbles_eaten

    @property
    def marble_data_eaten(self) -> List[object]:
        return self.__marble_data_eaten.copy()

    def eat(self, marble: Marble):
        self.__num_marbles_eaten += 1
        self.__marble_data_eaten.append(marble.datum)

    def copy(self) -> MarbleEaterNode:
        """
        Create copy of this MarbleEaterNode,
        but reset the number of marbles eaten of the copy to 0.
        """
        output = MarbleEaterNode(self.init_pos,
                               self.init_vel,
                               self.init_acc,
                               self.mass,
                               self._attraction_function,
                               self.marble_stiffness,
                               self.node_stiffness,
                               self.marble_attraction,
                               self.node_attraction,
                               self.radius)
        output.adopt_parameters(self)
        return output

    def reset(self):
        super().reset()
        self.num_marbles_eaten = 0
        self.__marble_data_eaten = []
