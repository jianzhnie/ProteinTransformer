from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

import os


class SkipLSTM(nn.Module):

    def __init__(self,
                 inputs_size,
                 output_size,
                 hidden_size,
                 num_layers,
                 bidirectional=True,
                 dropout=0,
                 **kwargs):
        super(SkipLSTM, self).__init__()

        self.input_size = inputs_size
        self.output_size = output_size

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        dim = inputs_size
        for _ in range(num_layers):
            f = nn.LSTM(inputs_size=dim,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        batch_first=True,
                        bidirectional=bidirectional,
                        **kwargs)
            self.layers.append(f)
            if bidirectional:
                dim = 2 * hidden_size
            else:
                dim = hidden_size

        num_features = hidden_size * num_layers + inputs_size
        if bidirectional:
            num_features = 2 * hidden_size * num_layers + inputs_size

        self.proj = nn.Linear(num_features, output_size)

    def to_one_hot(self, x):
        packed = type(x) is PackedSequence
        if packed:
            one_hot = x.data.new(x.data.size(0),
                                 self.input_size).float().zero_()
            one_hot.scatter_(1, x.data.unsqueeze(1), 1)
            one_hot = PackedSequence(one_hot, x.batch_sizes)
        else:
            one_hot = x.new(x.size(0), x.size(1),
                            self.input_size).float().zero_()
            one_hot.scatter_(2, x.unsqueeze(2), 1)
        return one_hot

    def transform(self, x):
        one_hot = self.to_one_hot(x)
        hs = [one_hot]  # []
        h_ = one_hot
        for f in self.layers:
            h, _ = f(h_)
            hs.append(h)
            h_ = h
        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            h = PackedSequence(h, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
        return h

    def forward(self, x):
        one_hot = self.to_one_hot(x)
        hs = [one_hot]
        h_ = one_hot

        for f in self.layers:
            h, _ = f(h_)
            hs.append(h)
            h_ = h

        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            z = self.proj(h)
            z = PackedSequence(z, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
            z = self.proj(h.view(-1, h.size(2)))
            z = z.view(x.size(0), x.size(1), -1)

        return z