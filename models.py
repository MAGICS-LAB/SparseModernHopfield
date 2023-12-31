import torch
import torch.nn as nn

from layers import *

class MNISTModel(nn.Module):
    def __init__(
            self,
            input_size,
            d_model,
            n_heads,
            update_steps,
            dropout,
            mode,
            scale=None,
            num_pattern=1):
        super(MNISTModel, self).__init__()

        assert d_model % n_heads == 0
        self.emb = nn.Linear(input_size, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        if mode in ["sparsemax", 'softmax', 'entmax', 'gsh']:
            self.layer = HopfieldPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                dropout=dropout,
                mode=mode,
                scale=scale,
                num_pattern=num_pattern)

        self.fc = nn.Linear(d_model*num_pattern, 1)
        self.gelu = nn.GELU()

    def forward(self, x):

        bz, N, c, h, w = x.size()
        x = x.view(bz, N, -1)
        x = self.ln(self.emb(x))
        out = self.ln2(self.gelu(self.layer(x)))
        out = out.view(bz, -1)
        return self.fc(out).squeeze(-1)
    



class CIFARModel(nn.Module):
    def __init__(
            self,
            d_model=256,
            n_heads=4,
            update_steps=1,
            dropout=0.1,
            mode='softmax',
            scale=None,
            num_pattern=1):
        super(CIFARModel, self).__init__()

        assert d_model % n_heads == 0
        self.emb = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(4, 2)
        self.linear = nn.Linear(1014, d_model)

        self.ln = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        if mode in ["sparsemax", 'softmax', 'entmax', 'gsh']:
            self.layer = HopfieldPooling(
                d_model=d_model,
                n_heads=n_heads,
                mix=True,
                update_steps=update_steps,
                dropout=dropout,
                mode=mode,
                scale=scale,
                num_pattern=num_pattern)

        self.fc = nn.Linear(d_model*num_pattern, 1)
        self.gelu = nn.GELU()

    def forward(self, x):

        bz, N, c, h, w = x.size()
        x = x.view(bz*N, c, h, w)
        x = self.pool(self.emb(x))
        x = x.view(bz, N, -1)
        x = self.ln(self.linear(x))
        out = self.ln2(self.gelu(self.layer(x)))
        out = out.view(bz, -1)
        return self.fc(out).squeeze(-1)