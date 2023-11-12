import numpy as np
import scipy.io
import os
import pickle
import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold

import torch
import torch.utils.data
from torchvision import datasets, transforms

from .loader_utils import *
import random

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()

        self.x = x
        self.y = y
        for i in range(len(self.y)):
            if self.y[i] == -1:
                self.y[i] = 0.0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        batch_x = self.x[idx] # (bag_size, feat_dim)
        batch_y = self.y[idx]

        batch_x = torch.tensor(batch_x)
        batch_y = torch.tensor(batch_y)

        return batch_x, batch_y

    def collate(self, batch):

        x = [x for x,y in batch]
        y = [y for x,y in batch]

        pad_batch_x, mask_x = self.padding(x)

        return pad_batch_x, torch.stack(y, dim=0), mask_x

    def padding(self, batch):

        max_bag_len = max([len(xi) for xi in batch]) # (batch_size, bag_size, feat_dim)
        feat_dim = batch[0].size(-1)

        batch_x_tensor = torch.zeros((len(batch), max_bag_len, feat_dim))
        mask_x = torch.ones((len(batch), max_bag_len), dtype=torch.uint8)

        for i in range(len(batch)):
            bag_size = batch[i].size(0)
            batch_x_tensor[i, :bag_size] = batch[i]
            mask_x[i][:bag_size] = 0.0

        mask_x = mask_x.to(torch.bool)
        return batch_x_tensor, mask_x

def load_data(args):
    features = []
    labels = []
    current_file = os.path.abspath(os.path.dirname(__file__))
    dataset = scipy.io.loadmat(current_file + '/mil_datasets/{args.dataset}_100x100_matlab.mat')  # loads fox dataset
    instance_bag_ids = np.array(dataset['bag_ids'])[0]
    instance_features = np.array(dataset['features'].todense())
    if args.multiply:
        instance_features = multiply_features(instance_features)

    instance_labels = np.array(dataset['labels'].todense())[0]
    bag_features = into_dictionary(instance_bag_ids,
                                   instance_features)  # creates dictionary whereas key is bag and values are instance
    bag_labels = into_dictionary(instance_bag_ids,
                                 instance_labels)  # creates dictionary whereas key is bag and values are instance
    for i in range(1, len(bag_features) + 1):  # goes through whole dataset
        features.append(np.array(bag_features.pop(i)))
        labels.append(max(bag_labels[i]))
    return features, labels

def load_ucsb():
    
    '''
    This function Returns ussb bag features and bag labels
    '''

    def load_ucsb_data(filepath):
        df = pd.read_csv(filepath, header=None)
        
        bags_id = df[1].unique()
        bags = [df[df[1]==bag_id][df.columns.values[2:]].values.tolist() for bag_id in bags_id]
        y = df.groupby([1])[0].first().values
        bags = [np.array(b) for b in bags]
        return bags, np.array(y)

    current_file = os.path.abspath(os.path.dirname(__file__))
    return load_ucsb_data(current_file + '/csv/ucsb_breast_cancer.csv')
