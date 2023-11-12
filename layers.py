import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import math
from math import sqrt
from utils.sparse_max import Sparsemax
from utils.entmax import Entmax15


class FullAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, attention_dropout=0.0):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, H, scores.size(-2), 1)
            scores = scores.masked_fill_(mask, float('-inf'))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()

class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=True,
            dropout=0.1,
            scale=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model
        self.inner_attention = FullAttention(
            scale=scale, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, inputs):

        queries = inputs
        keys = inputs
        values = inputs

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        out = out.view(B, L, -1)
        out = out.mean(1)

        return self.out_projection(out)

class HopfieldCore(nn.Module):
    '''
    The Hopfield operation
    '''

    def __init__(self, scale=None, attention_dropout=0.0, mode='sparsemax', norm=False):
        super(HopfieldCore, self).__init__()
        self.scale = scale
        self.norm = norm
        self.dropout = nn.Dropout(attention_dropout)
        if mode == 'sparsemax':
            self.softmax = Sparsemax(dim=-1)
        elif mode == 'entmax':
            self.softmax = Entmax15(dim=-1)
        else:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values, mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.norm and H == 1:
            scores = F.normalize(scores)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, H, scores.size(-2), 1)
            scores = scores.masked_fill_(mask, float('-inf'))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()

class Hopfield(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=True,
            update_steps=1,
            dropout=0.1,
            mode='softmax',
            scale=None):
        super(Hopfield, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model

        self.inner_attention = HopfieldCore(
            scale=scale, attention_dropout=dropout, mode=mode)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(
            d_values * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.update_steps = update_steps

    def forward(self, R, Y, mask=None):

        B, L, _ = R.shape
        _, S, _ = Y.shape
        H = self.n_heads

        queries = self.query_projection(R).view(B, L, H, -1)
        keys = self.key_projection(Y)
        values = self.value_projection(keys).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)

        for i in range(self.update_steps):

            queries = self.inner_attention(
                queries,
                keys,
                values,
                mask
            )

        out = queries
        out = out.view(B, L, -1)

        return self.out_projection(out)

class HopfieldPooling(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=True,
            num_pattern=1,
            update_steps=1,
            dropout=0.1,
            mode='softmax',
            scale=None):
        super(HopfieldPooling, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model

        self.inner_attention = HopfieldCore(
            scale=scale, attention_dropout=dropout, mode=mode)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(
            d_values * n_heads, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.update_steps = update_steps

        pooling_weight_size = d_model

        self.query = nn.Parameter(
            torch.randn(
                size=(
                    *
                    (
                        (1,
                         num_pattern)),
                    d_model if pooling_weight_size is None else pooling_weight_size), dtype=torch.float32),
            requires_grad=True)

    def forward(self, Y, mask=None):

        # queries : state pattern
        # keys : store pattern
        # values : should just be keys
        B, L, _ = self.query.shape
        B, S, _ = Y.shape
        H = self.n_heads
        q = self.query.repeat((*((B, 1)), 1))

        queries = self.query_projection(q).view(B, L, H, -1)
        keys = self.key_projection(Y)
        values = self.value_projection(keys).view(B, S, H, -1)
        keys = keys.view(B, S, H, -1)

        for _ in range(self.update_steps):

            queries = self.inner_attention(
                queries,
                keys,
                values,
                mask
            )

        out = queries
        out = out.view(B, L, -1)
        return self.out_projection(out)

class HopfieldLayer(nn.Module):
    def __init__(
            self,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
            mix=False,
            update_steps=1,
            dropout=0.0,
            mode='softmax',
            scale=None):
        super(HopfieldLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model

        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)

        if mode in ["sparsemax", "softmax", "entmax"]:
            self.inner_attention = HopfieldCore(
                scale=scale, attention_dropout=dropout, mode=mode, norm=True)

        self.n_heads = n_heads
        self.mix = mix
        self.update_steps = update_steps

    def forward(self, R, Y):

        # R : query pattern
        # Y : memory pattern

        B, L, _ = R.shape
        B, S, _ = Y.shape
        H = self.n_heads
        # R, Y = self.ln(R), self.ln(Y)

        queries = R.view(B, L, H, -1)
        keys = Y.view(B, S, H, -1)
        values = Y.view(B, S, H, -1)

        for _ in range(self.update_steps):

            queries = self.inner_attention(
                queries,
                keys,
                values,
            )

        out = queries
        out = out.view(B, L, -1)
        return out
 
