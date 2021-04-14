"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Teun Schilperoort, Lulof Pirée
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


Auxiliary function for loading the MNIST dataset
and splitting it into a train, test and validation set.
reference: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""
from typing import Iterable, Iterator, Sequence, Tuple
import torch
from torchvision import datasets, transforms
import torchvision
import torch.utils
import math
import random
import os
import pickle

from nenwin.backprop.dataset import Dataset, Sample
from nenwin.constants import MNIST_DATA_DIR, MNIST_CACHE_FILE


class MNISTDataset(Dataset):

    def __init__(self,
                 train_dataset: Sequence[Sample],
                 vali_dataset: Sequence[Sample],
                 test_dataset: Sequence[Sample]):
        self.__train_data = train_dataset
        self.__vali_dataset = vali_dataset
        self.__test_dataset = test_dataset

    def iter_train(self) -> Iterator[Sample]:
        return iter(random.sample(self.__train_data, len(self.__train_data)))

    def iter_validation(self) -> Iterator[Sample]:
        return iter(self.__vali_dataset)

    def iter_test(self):
        return iter(self.__test_dataset)

    def get_len_train(self) -> int:
        return len(self.__train_data)

    def get_len_validation(self) -> int:
        return len(self.__vali_dataset)

    def get_len_test(self) -> int:
        return len(self.__test_dataset)

    def __str__(self) -> str:
        output = (f"MNISTDataset\n\ttrain samples: {self.get_len_train()}"
                  f"\n\tvali samples: {self.get_len_validation()}"
                  f"\n\ttest samples: {self.get_len_test()}"
                  f"\n\tinput type: {type(self.__train_data[0].inputs)}"
                  f"\n\tlabel type: {type(self.__train_data[0].label)}")
        return output


def load_mnist_dataset() -> MNISTDataset:
    if os.path.exists(MNIST_CACHE_FILE):
        with open(MNIST_CACHE_FILE, "rb") as file:
            dataset = pickle.load(file)
    else:
        dataset = convert_mnist_dataset()
        with open(MNIST_CACHE_FILE, "wb") as file:
            pickle.dump(dataset, file)
    return dataset



def convert_mnist_dataset() -> MNISTDataset:
    """
    Load the MNIST dataset as a training-, vaidation- and test-set.
    The validation set are 10000 samples of the 'training set' (according to the
    predefined split. So the actually returned training set are the remaining
    50000 samples of the predefined 'training set').

    All inputs are transformed to Tensors with values in the
    range [0, 1]. The labels are integers, corresponding to the
    digits the input images are supposed to represent.
    """

    train_plus_vali = torchvision.datasets.MNIST(
        root=MNIST_DATA_DIR, train=True, transform=transforms.ToTensor(),
        download=True)
    test_dataset = torchvision.datasets.MNIST(root=MNIST_DATA_DIR,
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)
    print(test_dataset[0])
    # train_plus_vali, test_dataset, scaler = __scale_datasets(train_plus_vali,
    #                                                          test_dataset)
    test_samples = convert_to_samples(test_dataset)
    train_plus_vali_samples = convert_to_samples(train_plus_vali)

    num_train_samples = len(train_plus_vali)
    train_samples = train_plus_vali_samples[:50000]
    vali_samples = train_plus_vali_samples[50000:]

    return MNISTDataset(train_samples, vali_samples, test_samples)


def __scale_datasets(train_plus_vali: torch.utils.data.Dataset,
                     test_dataset: torch.utils.data.Dataset) -> tuple:
    max_value = torch.max(train_plus_vali.data)
    min_value = torch.min(train_plus_vali.data)
    print(max_value, max_value.shape)
    scaler = MinMaxScaler(max_value, min_value)
    train_plus_vali.data = scaler.apply_on(train_plus_vali.data)
    test_dataset.data = scaler.apply_on(test_dataset.data)

    return train_plus_vali, test_dataset, scaler


def convert_to_samples(dataset: torch.utils.data.Dataset) -> Tuple[Sample]:
    return tuple(map(lambda x: Sample(x[0], x[1]), dataset))


class MinMaxScaler:
    """
    Encapsulation of Min-Max Scaling, that stores the constant
    and denominator terms used in the scaling.
    """

    def __init__(self, max_value: float, min_value: float):
        self.max = max_value
        self.min = min_value

    def apply_on(self, dataset: torch.utils.data.Dataset
                 ) -> torch.utils.data.Dataset:
        """
        Apply min-max scaling (or ´normalization´ or ´standadization´)
        with the stored min and max values.
        """
        dataset = (dataset - self.min) / (self.max-self.min)
        return dataset

    def __call__(self, dataset: torch.utils.data.Dataset
                 ) -> torch.utils.data.Dataset:
        """
        Apply min-max scaling (or ´normalization´ or ´standadization´)
        with the stored min and max values.
        """
        return self.apply_on(dataset)


if __name__ == "__main__":
    # print(MNIST_DATA_DIR)
    print(load_mnist_dataset())
