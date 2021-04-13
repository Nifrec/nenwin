"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
April 2021

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

from nenwin.backprop.dataset import Dataset, Sample
from typing import Any, List, Sequence, Optional, Iterable, Tuple, Iterator
import torch
import unittest
import numpy as np

from nenwin.backprop.trainer import NenwinTrainer
from nenwin.backprop.loss_function import NenwinLossFunction
from nenwin.input_placer import InputPlacer, ZERO
from nenwin.backprop.filename_gen import FilenameGenerator
from nenwin.model import NenwinModel
from nenwin.all_particles import Marble, MarbleEaterNode

DUMMY_MARBLE = Marble(ZERO, ZERO, ZERO, 0, None, None)


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
        self.set_outputs(outputs)
        if expected_targets is not None:
            self.set_targets(expected_targets)

        self._NenwinLossFunction__output_nodes = []

    def __call__(self, expected: int) -> torch.Tensor:
        if self.expected_targets is not None:
            assert expected == self.expected_targets.pop()
        return torch.tensor([1.0], requires_grad=True) * self.outputs.pop()

    def set_targets(self, expected_targets: Sequence[int]):
        self.expected_targets = list(reversed(expected_targets))

    def set_outputs(self, outputs: Sequence[float]):
        self.outputs = list(reversed(outputs))


class MockOptimizer():

    def step(self):
        ...

    def zero_grad(self):
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

    def __repr__(self) -> str:
        return f"MockModel({repr(self.eaters)})"


class MockInputPlacer(InputPlacer):

    def __init__(self):
        ...

    def marblelize_data(self, input_data: Iterable[object]) -> tuple:
        return tuple()


class MockDataset(Dataset):

    def __init__(self, train_samples: Iterable[Sample] = [],
                 vali_samples: Iterable[Sample] = [],
                 test_samples: Iterable[Sample] = []):
        self.trainset = train_samples
        self.valiset = vali_samples
        self.testset = test_samples

    def iter_train(self) -> Iterator[Sample]:
        return iter(self.trainset)

    def get_len_train(self) -> int:
        return len(self.trainset)

    def iter_validation(self) -> Iterator[Sample]:
        return iter(self.valiset)

    def get_len_validation(self) -> int:
        return len(self.valiset)

    def iter_test(self) -> Iterator[Sample]:
        return iter(self.testset)

    def get_len_test(self) -> int:
        return len(self.testset)


class NenwinTrainerTestCase(unittest.TestCase):

    def setUp(self):
        self.eaters: Tuple[MarbleEaterNode] = (
            MarbleEaterNode(ZERO, ZERO, ZERO, 0, None, 0, 0, 0, 0, 0),
        ) * 6

        self.model = MockModel(self.eaters)

        self.file_gen = FilenameGenerator(".", "TEST__", "")

        dataset = self.set_up_dataset()

        self.train_losses = (1, 2, 3)
        self.vali_losses = (3, 4, 5)
        self.test_losses = (6, 7, 8)

        self.loss_funct = MockLossFunction(
            self.train_losses + self.vali_losses,
            self.train_targets + self.vali_targets)

        
        self.trainer = NenwinTrainer(self.model, self.loss_funct,
                                     MockOptimizer(), self.file_gen,
                                     MockInputPlacer(), self.dataset)

    def set_up_dataset(self) -> Dataset:
        self.train_targets = (1, 2, 3)
        trainset = [Sample(value, label) for (value, label)
                    in zip([None]*3, self.train_targets)]

        self.vali_targets = (3, 4, 5)
        valiset = [Sample(value, label) for (value, label)
                   in zip([None]*3, self.vali_targets)]

        
        self.test_targets = (0, 1, 1)
        testset = [Sample(value, label) for (value, label)
                   in zip([None]*3, self.test_targets)]

        self.dataset = MockDataset(trainset, valiset, testset)


    def test_reset_stats(self):
        self.trainer.run_training(1, self.dataset, True)
        self.trainer.reset_stats()

        result = self.trainer.training_stats

        empty = tuple()
        self.assertTupleEqual(result.train_losses, empty)
        self.assertTupleEqual(result.validation_accuracies, empty)
        self.assertTupleEqual(result.validation_losses, empty)



    def test_evaluate_model_on_dataset_case_vali(self):
        """
        Evaluating on the *validation* set should 
        return the correct accuracy and loss.
        """
        self.loss_funct.set_outputs(self.vali_losses)
        self.loss_funct.set_targets(self.vali_targets)

        # Make model output [0, 0, 0, 1, 0, 0]
        output = 3
        self.eaters[output].eat(DUMMY_MARBLE)
        expected_loss = np.mean(self.vali_losses)
        expected_acc = np.sum(np.array(self.vali_targets) == output)
        result_acc, result_loss = self.trainer.evaluate_model(True)

        self.assertAlmostEqual(expected_loss, result_loss)
        self.assertAlmostEqual(expected_acc, result_acc)

        
    def test_evaluate_model_on_dataset_case_test(self):
        """
        Evaluating on the *test* set should 
        return the correct accuracy and loss.
        """
        self.loss_funct.set_outputs(self.test_losses)
        self.loss_funct.set_targets(self.test_targets)

        # Make model output [0, 1, 0, 0, 0, 0]
        output = 1
        self.eaters[output].eat(DUMMY_MARBLE)
        expected_loss = np.mean(self.test_losses)
        expected_acc = np.sum(np.array(self.test_targets) == output)
        result_acc, result_loss = self.trainer.evaluate_model(False)

        self.assertAlmostEqual(expected_loss, result_loss)
        self.assertAlmostEqual(expected_acc, result_acc)


    def test_evaluate_model_not_affect_stats(self):
        """
        Evaluating should not affect the training stats maintained
        by the NenwinTrainer.
        """
        self.loss_funct.set_outputs(self.test_losses)
        self.loss_funct.set_targets(self.test_targets)
        self.eaters[0].eat(DUMMY_MARBLE)
        self.trainer.evaluate_model(False)

        result = self.trainer.training_stats

        self.assertTupleEqual(result.train_losses, tuple())
        self.assertTupleEqual(result.validation_accuracies, tuple())
        self.assertTupleEqual(result.validation_losses, tuple())

    def test_run_training_with_validation(self):
        """
        When running the training with validation enabled,
        the mean epoch loss should be stored for the validation 
        and the train set.
        In addition, also the mean epoch accuracy of the validation
        set should be stored.
        """
        output = 3
        # Make model output [0, 0, 0, 1, 0, 0]
        self.eaters[output].eat(DUMMY_MARBLE)
        num_iters = 1

        self.trainer.run_training(num_iters, 1, 1, True)
        self.trainer.reset_stats()

        result = self.trainer.training_stats

        expected_train_losses = (np.mean(self.train_losses),)*num_iters
        expected_vali_losses = (np.mean(self.vali_losses),)*num_iters
        expected_vali_accs = (
            np.sum(np.array(self.vali_targets) == output),) * num_iters

        np.testing.assert_allclose(result.train_losses, expected_train_losses)
        self.assertTupleEqual(result.validation_accuracies, expected_vali_accs)
        self.assertTupleEqual(result.validation_losses, expected_vali_losses)

    def test_run_training_without_validation(self):
        """
        When running the training with validation *disabled*,
        only the mean epoch loss of the train set should be recorded.
        """
        num_iters = 3

        self.trainer.run_training(num_iters, 1, 1, False)
        self.trainer.reset_stats()

        result = self.trainer.training_stats

        expected_train_losses = (np.mean(self.train_losses),)*num_iters

        np.testing.assert_allclose(result.train_losses, expected_train_losses)
        self.assertTupleEqual(result.validation_accuracies, tuple())
        self.assertTupleEqual(result.validation_losses, tuple())

    def test_get_current_model_output_case_none(self):
        """
        If none of the output eaters did eat a Marble,
        Trainer.get_current_model_output() should return None;
        """
        expected = None
        result = self.trainer.get_current_model_output()
        self.assertEqual(expected, result)

    def test_get_current_model_output_base_case(self):
        """
        Base case: index of the only MarbleEaterNode
        of the output-eaters should be returned by 
        Trainer.get_current_model_output().
        """
        # Make model output [0, 0, 0, 1, 0, 0]
        self.eaters[3].eat(DUMMY_MARBLE)
        expected = 3
        result = self.trainer.get_current_model_output()
        self.assertEqual(expected, result)


    def test_get_current_model_output_muli_output_case(self):
        """
        In case of multiple outputs,
        Trainer.get_current_model_output()
        should return a sorted tuple with the indices of
        all activated output-eaters.
        """
        # Make model output [0, 1, 0, 0, 0, 1]
        self.eaters[1].eat(DUMMY_MARBLE)
        self.eaters[5].eat(DUMMY_MARBLE)

        expected = (1, 5)

        result = self.trainer.get_current_model_output()
        self.assertTupleEqual(expected, result)


if __name__ == "__main__":
    unittest.main()
