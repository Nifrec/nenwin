"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Extension of NenwinModel that behaves the same,
but also counts down the delays of any Emitters carried by
EmitterNodes, and adds emitted Marbles to the model
when possible.
"""
import numpy as np
from typing import Iterable, Optional

from experiment_1.model import NenwinModel
from experiment_1.marble_emitter_node import MarbleEmitterNode
from experiment_1.node import Marble, Node


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


