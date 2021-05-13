"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Copyright (C) 2020 Lulof Pirée, 

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

Extension of NenwinModel that behaves the same,
but also counts down the delays of any Emitters carried by
EmitterNodes, and adds emitted Marbles to the model
when possible.
"""
import numpy as np
from typing import Iterable, Optional

from nenwin.model import NenwinModel
from nenwin.marble_emitter_node import MarbleEmitterNode
from nenwin.node import Marble, Node


class ModelWithEmitters(NenwinModel):
    """
    Extension of NenwinModel that behaves the same,
    but also counts down the delays of any Emitters carried by
    EmitterNodes, and adds emitted Marbles to the model
    when possible.
    """

    def __init__(self,
                 nodes: Iterable[Node],
                 initial_marbles: Optional[Iterable[Marble]] = None):
        super().__init__(nodes, initial_marbles)
        self.__emitters = set(node.emitter for node in self.nodes
                                if isinstance(node, MarbleEmitterNode))

    def make_timestep(self, time_passed: float):
        """
        Make step similar to NenwinModel,
        but also update the delays of the emitters,
        and emit new Marbles where needed.
        """
        new_marbles = set()
        for emitter in self.__emitters:
            emitter.register_time_passed(time_passed)
            if emitter.can_emit():
                new_marbles.add(emitter.emit())
            
        self.add_marbles(new_marbles)
        super().make_timestep(time_passed)


