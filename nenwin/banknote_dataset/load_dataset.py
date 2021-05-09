"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author:Lulof Pirée
May 2021

Copyright (C) 2021 Lulof Pirée

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


Functions and classes for loading the banknote dataset
into a Dataset instance.

Dataset source: https://code.datasciencedojo.com/datasciencedojo/datasets/blob/master/Banknote%20Authentication/data_banknote_authentication.txt
See also:
* https://jamesmccaffrey.wordpress.com/2020/08/18/in-the-banknote-authentication-dataset-class-0-is-genuine-authentic/
* https://www.researchgate.net/publication/266673146_Banknote_Authentication
This dataset has:
* 1372 samples, of two classes:
    * class 0 (Genuine): 762 samples 
    * class 1 (Forgery): 610 samples

Features:
0: variance 
1: skewness
2: curtosis
3: entropy
"""
from typing import Iterable, Iterator, Sequence, Tuple
import torch
import torch.utils
import math
import random
import os
import pickle
import pandas as pd
import numpy as np

from nenwin.backprop.dataset import Dataset, Sample
from nenwin.constants import BANKNOTE_DATA_FILE, BANKNOTE_CACHE_FILE


class BanknoteDataset(Dataset):

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
        output = (f"BanknoteDataset\n\ttrain samples: {self.get_len_train()}"
                  f"\n\tvali samples: {self.get_len_validation()}"
                  f"\n\ttest samples: {self.get_len_test()}"
                  f"\n\tinput type: {type(self.__train_data[0].inputs)}"
                  f"\n\tlabel type: {type(self.__train_data[0].label)}")
        return output


def load_banknote_dataset() -> BanknoteDataset:
    if os.path.exists(BANKNOTE_CACHE_FILE):
        with open(BANKNOTE_CACHE_FILE, "rb") as file:
            dataset = pickle.load(file)
    else:
        dataset = convert_banknote_dataset()
        with open(BANKNOTE_CACHE_FILE, "wb") as file:
            pickle.dump(dataset, file)
    return dataset


def convert_banknote_dataset(frac_train=0.8, frac_vali=0.1) -> BanknoteDataset:
    """
    Load the Banknote dataset as a training-, vaidation- and test-set.

    Arguments:
    * frac_train: relative fraction of samples to be used as the training set.
    * frac_vali: relative fraction of samples to be used as validation set.

    The remaining samples will constitute the test set.
    """
    assert frac_train + frac_vali <= 1.0

    df = pd.read_csv(BANKNOTE_DATA_FILE)
    # Shuffle the rows of the dataset. By default, it is sorted by class.
    df = df.sample(frac=1).reset_index(drop=True)

    tot_num_samples = len(df)
    first_vali_idx = np.floor(frac_train*tot_num_samples)
    first_test_idx = np.floor((frac_train + frac_vali)*tot_num_samples)

    samples = convert_to_samples(df)

    train_set = samples[:first_vali_idx]
    vali_set = samples[first_vali_idx:first_test_idx]
    test_set = samples[first_test_idx:]

    return BanknoteDataset(train_set, vali_set, test_set)


def convert_to_samples(df: pd.DataFrame) -> Tuple[Sample]:
    return tuple(Sample(row[1][:-1].to_numpy(), int(row[1][-1]))
                 for row in df.iterrows())
