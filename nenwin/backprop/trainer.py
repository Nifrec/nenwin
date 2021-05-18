"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
May 2021

Copyright (C) 2021 Lulof Pirée, 

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


Class for managing the training of a NenwinModel using backpropagation.
"""
from __future__ import annotations
import torch
import matplotlib.axes as axes
import matplotlib.pyplot as plt
from typing import Callable, Iterable, Optional, Sequence, Tuple
import time
import os

from nenwin.backprop.loss_function import NenwinLossFunction
from nenwin.model import NenwinModel
from nenwin.backprop.training_stats import TrainingStats
from nenwin.input_placer import InputPlacer
from nenwin.backprop.filename_gen import FilenameGenerator
from nenwin.backprop.dataset import Dataset


class NenwinTrainer:
    """
    Class for managing the training of a NenwinModel using backpropagation.
    """

    def __init__(self,
                 model: NenwinModel,
                 loss_funct: NenwinLossFunction,
                 optimizer: torch.optim.Optimizer,
                 filename_gen: FilenameGenerator,
                 input_placer: InputPlacer,
                 dataset: Dataset):
        """
        Arguments:
            * model: the Nenwin architecture to train.
            * loss_funct: loss function to optimize.
                Should be initialized with the given architecture
                beforehand if relevant.
            * optimizer: gradient-based optimization algorithm
                used to adjust the parameters of the model.
            * filename_gen: instance that generates
                filenames under which the trained model
                (and checkpoints) should be saved.
            * input_placer: mapping from a sample of the data
                to a set of Marbles.
            * dataset: collection of a training-, validation-
                and test-set of Samples 
                (numerical input vector and integer label).
        """
        self.__model = model
        self.__loss_funct = loss_funct
        self.__optim = optimizer
        self.__filename_gen = filename_gen
        self.__input_placer = input_placer
        self.__dataset = dataset
        self.__stats = TrainingStats()

    def run_training(self,
                     num_epochs: int,
                     step_size: float,
                     num_steps_till_read_output: int,
                     do_validate: bool = False,
                     checkpoint_interval: int = 1,
                     batch_size:int=1):
        """
        Arguments:
            * num_epochs: amount of full iterations of the training set
                should be ran (to train the model).
            * step_size: amount of time to pass between discrete
                simulation time-steps
            * num_steps_till_read_output: amount of timesteps after
                inputting a new sample after which the output
                should be read.
            * do_validate: if True, the model will be evaluated
                on one epoch on the validation set after each
                epoch after the training set. 
                These stats will be added to Trainer.stats.
            * checkpoint_interval: amount of epochs between
                making a checkpoint for the model.
                Making a checkpoint = storing the current state of the model
                to a file.
            * batch_size: amount of batches to complete
                (and gradients to accumulate) before
                performing a parameter update.
        """
        if batch_size > 1:
            for param in self.__model.parameters():
                param.register_hook(lambda grad : grad/batch_size)

        for epoch in range(num_epochs):

            epoch_loss = self.__run_one_trainset_epoch(
                step_size, 
                num_steps_till_read_output,
                batch_size)
            self.__stats.add_train_loss(epoch_loss)

            if do_validate:
                self.__run_and_record_one_validation_epoch(
                    step_size, num_steps_till_read_output)

            self.__make_checkpoint_if_needed(epoch, checkpoint_interval)

            print(f"Finished epoch {epoch}: loss = {epoch_loss}")

        print(f"Last epoch {epoch} finished: saving model...")
        filename = self.__save_model(False)
        print(f"Model saved as {filename}")

    def __run_one_trainset_epoch(self, step_size: float,
                                 max_timesteps: int,
                                 batch_size:int=1) -> float:
        """
        Returns the total loss.

        Arguments:
            * step_size: amount of time between discrete-time simulation
                steps.
            * max_timesteps: maximum amount of simulated timesteps
                per sample. If no output occurred after max_timesteps,
                the loss will be computed for the empty output.
            * batch_size: amount of batches to complete
                (and gradients to accumulate) before
                performing a parameter update.

        """

        epoch_loss = 0
        batch_idx = 0
        for sample in self.__dataset.iter_train():
            self.__model.reset()
            marbles = self.__input_placer.marblelize_data(sample.inputs)
            self.__model.add_marbles(marbles)

            try:
                for _ in range(max_timesteps):
                    self.__model.make_timestep(step_size)
                    if self.get_current_model_output() is not None:
                        break
                
                loss = self.__loss_funct(sample.label)
                loss.backward()

                batch_idx += 1
                if (batch_idx+1)%batch_size == 0:
                    self.__optim.step()
                    self.__optim.zero_grad()
                    self.__model.clamp_all_particles()

                    epoch_loss += loss.item()
            except Exception as e:
                print(f"Skipping epoch due to error: {e}")

        self.__model.reset()
        return epoch_loss

    def __run_and_record_one_validation_epoch(self, step_size: float,
                                              num_steps_till_read_output: int):
        acc, val_loss = self.evaluate_model(
            step_size, num_steps_till_read_output, True)
        self.__stats.add_validation_accuracy(acc)
        self.__stats.add_validation_loss(val_loss)

    def __make_checkpoint_if_needed(self, epoch: int, checkpoint_interval: int):
        # +1, as epochs start counting at 0.
        if (epoch+1) % checkpoint_interval == 0:
            print(f"Epoch {epoch}: saving model...")
            filename = self.__save_model(True)
            print(f"Model saved as {filename}")

    def __save_model(self, is_checkpoint: bool) -> str:
        filename = self.__filename_gen.gen_filename(is_checkpoint)
        with open(filename, "w") as file:
            file.write(repr(self.__model))
            file.write("\n# Num finished epochs: "
                       f"{len(self.__stats.train_losses)}")
        return filename

    def evaluate_model(self,
                       step_size: float,
                       num_steps_till_read_output: int,
                       use_validation: bool = False,
                       ) -> Tuple[float, float]:
        """
        Evaluate the current performance of the model
        on one epoch of the test set.
        Alternatively the validation set can be used instead.

        Arguments:
            * use_validation: flag if the validation set should be used.
                True -> validation set is used.
                False -> test set is used.

        Returns:
            * accuracy: fraction of correct predictions
            * loss: sum of losses of each prediction.
        """
        with torch.no_grad():
            if use_validation:
                dataset_iter = self.__dataset.iter_validation()
                dataset_size = self.__dataset.get_len_validation()
            else:
                dataset_iter = self.__dataset.iter_test()
                dataset_size = self.__dataset.get_len_test()

            num_correct = 0
            tot_loss = 0
            for sample in dataset_iter:
                self.__model.reset()
                marbles = self.__input_placer.marblelize_data(sample.inputs)
                self.__model.add_marbles(marbles)

                for _ in range(num_steps_till_read_output):
                    self.__model.make_timestep(step_size)

                if self.get_current_model_output() == sample.label:
                    num_correct += 1

                tot_loss += self.__loss_funct(sample.label).item()

            acc = num_correct/dataset_size

            self.__model.reset()
            return acc, tot_loss

    def reset_stats(self):
        self.__stats.reset()

    @property
    def training_stats(self) -> TrainingStats:
        """
        Return a TrainingStats instance where:
            * train_losses are the sums of the losses per epoch.
            * validation_losses are the sums of the validation losses
                of one epoch of the validation-set,
                evaluated after each epoch of training.
            * validation_acc is the accuracy on the validation set,
                evaluated after each epoch of training.

        Some of these attributes may be empty, in case the training
        has not been run, has been run without validating,
        or the stats have been reset.
        """
        return self.__stats

    @property
    def model(self) -> NenwinModel:
        return self.__model

    def get_current_model_output(self) -> int | None | Tuple[int]:
        """
        Map the state of the NenwinModel to a classification prediction.
        Return None in case no Marbles 
        have been eaten by any of the output-eaters.
        """
        output_nodes = self.__loss_funct.output_nodes

        output_indices = []
        for node_idx in range(len(output_nodes)):
            node = output_nodes[node_idx]
            if node.num_marbles_eaten >= 1:
                output_indices.append(node_idx)

        if len(output_indices) == 1:
            return output_indices[0]
        elif len(output_indices) == 0:
            return None
        else:
            return tuple(output_indices)
