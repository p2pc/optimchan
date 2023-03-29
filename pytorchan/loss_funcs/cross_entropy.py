import torch
import torch.nn as nn

from typing import Union

class BCEWithLogitsLoss(nn.Module):
    """Binary-Cross Entropy with Logits Loss function
    """
    def __init__(self, **kwargs):
        """Initialization method forBCEWithLogitsLoss Class
        """
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def __call__(self, output, target):
        """Call method for perform loss computation.

        Args:
            output (torch.Tensor): Output tensor data.
            target (torch.Tensor): Target tensor data.
        """
        target = target.type_as(output)
        if len(target.shape) != len(output.shape):
            target = target.unsqueeze(1)
        return self.loss(output, target)


class WeightedBCEWithLogitsLoss(BCEWithLogitsLoss):
    """Weighted Binary-Cross Entropy with Logits Loss function
    """
    def __init__(self, beta: Union[float, int, list], **kwargs):
        """Initialization method for WeightedBCEWithLogitsLoss Class

        Args:
            beta (Union[float, int, list]): beta parameter
        """
        if isinstance(beta, (float, int)):
            self.beta = torch.Tensor([beta])

        if isinstance(beta, list):
            self.beta = torch.Tensor(beta)

        super().__init__(pos_weight=self.beta, **kwargs)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss function
    """
    def __init__(self, weight=None, **kwargs):
        """Initialization method for CrossEntropyLoss Class

        Args:
            weight (_type_, optional): Weight for loss function. Defaults to None.
        """
        if weight is not None:
            weight = torch.FloatTensor(weight)
        super().__init__(weight, **kwargs)