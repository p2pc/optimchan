import random

import numpy as np

try:
    import torch
except UserWarning:
    raise UserWarning('PyTorch has not already installed.')


def seed_everything(seed: int):
    """Seed everything for NumPy, PyTorch
    Args:
        seed (int): seed number for stochastic.
    Raises:
        UserWarning: PyTorch has not already installed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except UserWarning:
        raise UserWarning('PyTorch has not already installed.')

def seed_random(seed: int):
    """Seed everything for NumPy, PyTorch
    Args:
        seed (int): seed number for stochastic.
    """
    random.seed(seed)