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
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        x = torch.mm(input, self.weight)
        output = torch.spmm(adj, x)
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
