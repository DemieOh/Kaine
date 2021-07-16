import torch
import torch.nn as nn
import kaine.nn.function.metric.functional as F


class Metric(nn.Module):

    def __init__(self, reduction='mean'):
        super(Metric, self).__init__()
        self.reduction = reduction


class AngularError(Metric):
    
    __constants__ = ['reduction']
    
    def __init__(self, reduction: str = 'mean') -> None:
        super(AngularError, self).__init__(reduction=reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.angular_error(inputs, targets, reduction=self.reduction)
