"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import random


class CIFARBags(data_utils.Dataset):
    def __init__(
            self,
            target_number=9,
            bag_size=10,
            num_bag=500,
            pos_per_bag=1,
            seed=1,
            train=True):
        self.target_number = target_number
        self.bag_size = bag_size
        self.pos_per_bag = pos_per_bag
        self.train = train
        self.num_bag = num_bag

        self.r = np.random.RandomState(seed)

        self.num_in_train = 50000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if self.train:
            loader = data_utils.DataLoader(
                datasets.CIFAR10(
                    '../datasets',
                    train=True,
                    download=True,
                    transform=transform_train),
                batch_size=self.num_in_train,
                shuffle=False)
        else:
            loader = data_utils.DataLoader(
                datasets.CIFAR10(
                    '../datasets',
                    train=False,
                    download=True,
                    transform=transform_test),
                batch_size=self.num_in_test,
                shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        pos_idx = [i for i, j in enumerate(
            all_labels) if j == self.target_number]
        neg_idx = [i for i, j in enumerate(
            all_labels) if j != self.target_number]

        pos_images = []
        neg_images = []

        for i, img in enumerate(all_imgs):
            if all_labels[i] == self.target_number:
                pos_images.append(img)
            else:
                neg_images.append(img)

        self.all_pos_img = pos_images
        self.all_neg_img = neg_images

        for i in range(self.num_bag):

            _pos_idx = random.sample(pos_idx,
                                     self.pos_per_bag) + random.sample(neg_idx,
                                                                       self.bag_size - self.pos_per_bag)
            _neg_idx = random.sample(neg_idx, self.bag_size)
            assert len(_pos_idx) == len(_neg_idx)

            bags_list.append(all_imgs[_neg_idx])
            labels_list.append(1)
            bags_list.append(all_imgs[_pos_idx])
            labels_list.append(0)

        return bags_list, torch.tensor(labels_list)

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = self.train_labels_list[index]
        else:
            bag = self.test_bags_list[index]
            label = self.test_labels_list[index]

        return bag, label