import torch.nn as nn


class Metric(nn.Module):

    def __init__(self, reduction='mean'):
        super(Metric, self).__init__()
        self.reduction = reduction
