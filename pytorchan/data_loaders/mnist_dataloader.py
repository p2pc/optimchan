import torchvision

import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader


def get_dataset(data_path: str = '../datasets/', transform=None):
    train_dataset = torchvision.datasets.MNIST(
        data_path, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        data_path, train=False, transform=transform)
    return train_dataset, test_dataset


def loader(train_dataset, test_dataset, batch_size=128, test_size: float = 0.2):
    train_set_size = int(len(train_dataset) * (1-test_size))
    indices = list(range(train_set_size))
    split = int(np.floor(.2 * train_set_size))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SequentialSampler(val_indices)

    train_loader = DataLoader(dataset=train_dataset,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              )

    valid_loader = DataLoader(dataset=train_dataset,
                              sampler=valid_sampler,
                              batch_size=batch_size,
                              )

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)

    return train_loader, valid_loader, test_loader
