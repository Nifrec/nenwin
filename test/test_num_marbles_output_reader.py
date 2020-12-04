"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Unit-tests for NumMarblesOutputReader of output_reader.py.
"""
import numpy as np
import unittest
from typing import List
from experiment_1.model import NenwinModel
from experiment_1.output_reader import NumMarblesOutputReader
from experiment_1.node import Marble
from experiment_1.marble_eater_node import MarbleEaterNode
from test_aux import ZERO
from test_aux import check_close

class NumMarblesOutputReaderTestCase(unittest.TestCase):
    def setUp(self):
        self.output_reader = NumMarblesOutputReader()

    def test_output_1(self):
        """
        Base case: 3 Eater nodes with different eat counts.
        """
        marbles_eaten_per_node = [10, 20, 0]
        model = MockModel(marbles_eaten_per_node)
        result = self.output_reader.read_output(model)
        expected = np.array(marbles_eaten_per_node)
        self.assertTrue(check_close(result, expected))

    def test_output_2(self):
        """
        Corner case: no eater nodes in the model present.
        """
        marbles_eaten_per_node = []
        model = MockModel(marbles_eaten_per_node)
        result = self.output_reader.read_output(model)
        expected = np.array(marbles_eaten_per_node)
        self.assertTrue(check_close(result, expected))




class MockModel(NenwinModel):

    def __init__(self, marbles_eaten_per_node: List[int]):
        """
        Arguments:
        * marbles_eaten_per_node: list of amount of Marbles
            each MarbleEasterNode has eaten, in order as returned
            in MockModel.marble_eater_nodes. Ensures that
            MockModel.marble_eater_nodes returns Eaters with these 
            values of eaten Marbles in order.
        """
        self.marbles_eaten = marbles_eaten_per_node

    @property
    def marble_eater_nodes(self):
        output = []
        for marble_count in self.marbles_eaten:
            eater = MarbleEaterNode(ZERO, ZERO, ZERO, 0, None, 0, 0, 0, 0, 0)
            eater._MarbleEaterNode__num_marbles_eaten = marble_count
            output.append(eater)

        return output

if __name__ == '__main__':
    unittest.main()
