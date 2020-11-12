"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Class responsible for translating the state of a NenwinModel
to a interprentable output.
"""
import abc
from typing import Any
from experiment_1.model import NenwinModel

class OutputReader(abc.ABC):
    """
    Class responsible for translating the state of a NenwinModel
    to a interprentable output.
    """

    @abc.abstractmethod
    def read_output(self, model: NenwinModel) -> Any:
        """
        Translate current state of model to useful output.
        Abstract method to be defined by subclasses.
        """
        pass