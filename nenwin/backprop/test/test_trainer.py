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


Testcases for NenwinTrainer.
"""

from typing import Any, List, Sequence, Optional, Iterable, Tuple
import torch
import unittest
import numpy as np

from nenwin.backprop.trainer import NenwinTrainer
from nenwin.backprop.loss_function import NenwinLossFunction
from nenwin.input_placer import InputPlacer, ZERO
from nenwin.backprop.filename_gen import FilenameGenerator
from nenwin.model import NenwinModel
from nenwin.all_particles import Marble, MarbleEaterNode


class MockLossFunction(NenwinLossFunction):

    def __init__(self,
                 outputs: Sequence[torch.Tensor],
                 expected_targets: Optional[Sequence[int]] = None):
        """
        Arguments:
            * outputs: losses as returned in that order.
            * expected_targets: ordered expected indices of the correct class
                in loss computation.
                Everytime __call__ is called, it is checked if the
                target is the next target in this sequence.
        """
        self.outputs = list(reversed(outputs))
        if expected_targets is not None:
            self.expected_targets = list(reversed(expected_targets))

    def __call__(self, expected: int) -> torch.Tensor:
        if self.expected_targets is not None:
            assert expected == self.expected_targets.pop()
        return self.outputs.pop()


class MockOptimizer():

    def step():
        ...

    def zero_grad():
        ...


class MockModel(NenwinModel):

    def __init__(self, eaters: Sequence[MarbleEaterNode]):
        self.eaters = eaters
        

    def add_marbles(self, new_marbles: Iterable[Marble]):
        ...

    def make_timestep(self, time_passed: float):
        ...

    def reset(self):
        ...

    @property
    def marble_eater_nodes(self) -> List[MarbleEaterNode]:
        return self.eaters
        

class MockInputPlacer(InputPlacer):

    def marblize_data(self, input_data: Iterable[object]) -> tuple:
        return tuple()


class NenwinTrainerTestCase(unittest.TestCase):

    def setUp(self):
        self.eaters: Tuple[MarbleEaterNode] = (
            MarbleEaterNode(ZERO, ZERO, ZERO, 0, None, 0, 0, 0, 0)
        ) * 6
        self.eaters[3].eat(Marble(ZERO, ZERO, ZERO, 0, None))
        self.model = MockModel(self.eaters)
        self.loss_funct = MockLossFunction([1, 2, 3, 3, 4, 5],
                                           [1, 2, 3, 3, 4, 5])
        self.file_gen = FilenameGenerator(".", "TEST__", "")

        def dataset_callable():
            for i in zip(("input1", "input2", "input3"), (1, 2, 3)):
                yield i

        def valiset_callable():
            for i in zip(("val3", "val4", "val5"), (3, 4, 5)):
                yield i

        self.dataset_callable = dataset_callable
        self.valiset_callable = valiset_callable
        self.trainer = NenwinTrainer(self.model, self.loss_funct,
                                     MockOptimizer(), self.file_gen)

    def test_reset_stats(self):
        self.trainer.run_training(1,
                                  self.dataset_callable,
                                  self.dataset_callable)
        self.trainer.reset_stats()

        result = self.trainer.training_stats

        empty = tuple()
        self.assertTupleEqual(result.train_losses, empty)
        self.assertTupleEqual(result.validation_accuracies, empty)
        self.assertTupleEqual(result.validation_losses, empty)

    def test_visualize_model(self):
        self.fail()

    def test_eval_on_dataset(self):
        """
        Evaluating should return the correct accuracy and loss.
        """
        self.fail()

    def test_eval_on_dataset_not_affect_stats(self):
        """
        Evaluating should not affect the training stats maintained
        by the NenwinTrainer.
        """
        self.fail()

    def test_run_training_with_validation(self):
        num_iters = 1

        self.trainer.run_training(num_iters,
                                  self.dataset_callable,
                                  self.valiset_callable)
        self.trainer.reset_stats()

        result = self.trainer.training_stats

        expected_train_losses = \
            (torch.mean(torch.tensor([1, 2, 3.0])))*num_iters
        expected_vali_losses = \
            (torch.mean(torch.tensor((3, 4, 5))))*num_iters
        expected_vali_accs = (1/3)*num_iters

        np.testing.assert_allclose(result.train_losses, expected_train_losses)
        self.assertTupleEqual(result.validation_accuracies, expected_vali_accs)
        self.assertTupleEqual(result.validation_losses, expected_vali_losses)

    def test_run_training_without_validation(self):
        num_iters = 3

        self.trainer.run_training(num_iters,
                                  self.dataset_callable,
                                  self.dataset_callable)
        self.trainer.reset_stats()

        result = self.trainer.training_stats

        expected_train_losses = \
            (torch.mean(torch.tensor([1, 2, 3.0])))*num_iters

        np.testing.assert_allclose(result.train_losses, expected_train_losses)
        self.assertTupleEqual(result.validation_accuracies, tuple())
        self.assertTupleEqual(result.validation_losses, tuple())

    def get_current_model_output(self):
        ...

if __name__ == "__main__":
    unittest.main()
