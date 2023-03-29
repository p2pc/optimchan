import torchvision.transforms as transforms

from typing import List


def mnist_transform(lst_trans_operations: List = [transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                                  ]):
    return transforms.Compose(
        lst_trans_operations
    )
