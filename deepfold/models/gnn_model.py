import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, go_size, hidden_dimension):
        super().__init__()
        self.embed = nn.Linear(go_size, hidden_dimension)

    def forward(self, x):
        node_feature = self.embed(x)
        node_feature = F.normalize(node_feature)
        return node_feature


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""
    def __init__(self, input_dim, output_dim, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class CustomGCN(nn.Module):
    """A custom two layers GCN."""
    def __init__(self, nfeat, nhid, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.activate = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, adj):
        output = self.gc1(input, adj)
        output = self.activate(output)
        output = self.dropout(output)
        output = self.gc2(output, adj)
        output = self.activate(output)
        return output


class GCN(nn.Module):
    """A custom single layer GCN."""
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)

    def forward(self, input, adj):
        output = F.relu(self.gc1(input, adj))
        return output
