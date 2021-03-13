"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Teun Schilperoort
Maart 2021

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

Loading data used for training and testing the NENWIN model

reference: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           mean,
                           std,
                           valid_size=0.1,
                           num_workers=4,
                           pin_memory=False): 
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the dataset. pay attention to 
    num_workers (set to 2 if low-end computer, 4 or 6 if high-end computer)
    and pin_memory(False if CPU, True if GPU)
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    
    
def get_dataset(name: str = 'MNIST'):
    #define the mean and standard deviation for the dataset
    if name == 'MNIST':
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', 
                                                          download=True, 
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                          ])), 
                                           batch_size=10, 
                                           shuffle=True)
        
    else:
        print("not a valid dataset or not added yet")
        pass
    """
    Given a name of a dataset in torchvision and optional wei, returns (trainingset, testingset) as tuple. Normall
    """

