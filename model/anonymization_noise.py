from model.shift_gcn import Model as ActionClassifier
from model.shift_gcn import Model as PrivacyClassifier
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import sys
sys.path.insert(0, '')


class Anonymizer(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        N, C, T, V, M = x.size()  # (batch, xyz, frames, joints, #people)
        #mask = (x != 0).int()



        # (batch, xyz, frames, joints, #people)
        out = x + torch.normal(0, self.sigma, size=x.shape).to(x.device)
        #out *= mask

        return out


