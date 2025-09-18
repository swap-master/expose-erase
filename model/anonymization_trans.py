from model.shift_gcn import Model as ActionClassifier
from model.shift_gcn import Model as PrivacyClassifier
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
import sys
sys.path.insert(0, '')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Anonymizer(nn.Module):
    def __init__(self, num_point, num_person, graph=None, graph_args=dict(), in_channels=3):
        super().__init__()

        num_features = num_person * num_point * in_channels

        layernorm = nn.LayerNorm(256)
        encoderLayer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer=encoderLayer, num_layers=6, norm=layernorm)
        self.pe = PositionalEncoding(d_model=256, dropout=0)
        self.fc0 = nn.Linear(num_features, 256)
        self.fc1 = nn.Linear(256, num_features)

    def forward(self, x, is_mask=True):
        N, C, T, V, M = x.size()  # (batch, xyz, frames, joints, #people)
        mask = (x != 0).int()

        out = x
        # (batch, frames, xyz, joints, #people)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(N, T, C * V * M)

        x = self.fc0(x)
        x = self.pe(x)
        x = self.encoder(x)
        x = self.fc1(x)

        # (batch, xyz, frames, joints, #people)
        out = x.view(N, T, C, V, M).permute(0, 2, 1, 3, 4) + out
        if is_mask:
            out *= mask

        return out


class Model(nn.Module):
    def __init__(self,
                 num_privacy_class,
                 num_action_class,
                 num_point,
                 num_person,
                 graph,
                 graph_args,
                 in_channels=3):
        super().__init__()

        self.anonymizer = Anonymizer(
            num_point, num_person,
            graph, graph_args, in_channels)

    def load_action_classifier(self, path):
        self.action_classifier.load_state_dict(torch.load(path))

    def load_privacy_classifier(self, path):
        self.privacy_classifier.load_state_dict(torch.load(path))

    def forward(self, x):
        anonymized = self.anonymizer(x)

        privacy = self.privacy_classifier(anonymized)
        action = self.action_classifier(anonymized)

        return anonymized
