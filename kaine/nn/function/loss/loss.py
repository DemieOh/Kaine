import torch.nn as nn


class Loss(nn.Module):

    def __init__(self, reduction='mean'):
        super(Loss, self).__init__()
        self.reduction = reduction
