import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class ClassBalancedLosses(nn.Module):
    def __init__(self, nclass,
                 ignore_index=-1, beta=1 - 1e-3):
        super(ClassBalancedLosses, self).__init__()

        self.nclass = nclass
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(self, *inputs, ignore_index=-1):
        weights = self._class_balanced_weights(inputs[-1], self.nclass, self.beta).type_as(inputs[0])
        loss = F.cross_entropy(inputs[0], inputs[-1], weight=weights, ignore_index=-1)
        return loss

    @staticmethod
    def _class_balanced_weights(target, nclass, beta=1 - 1e-3):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = torch.zeros(batch, nclass)
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            # vect = hist>0
            tvect[i] = hist
        tvect_sum = torch.sum(tvect, 0)
        tvect_sum = (1 - beta) / (1 - beta ** (tvect_sum))
        tvect_sum[tvect_sum == np.inf] = 0
        return tvect_sum
