import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from .gnn_model import CustomGCN, Embedder


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.Linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        seq_feature = self.Linear(x)

        return seq_feature


class ContrastiveModel(nn.Module):
    def __init__(self,
                 seq_dim=1024,
                 num_nodes=40000,
                 node_feats=512,
                 hidden_dim=512,
                 num_labels=1000):
        super().__init__()
        self.num_labels = num_labels
        self.seq_mlp = MLP(seq_dim, hidden_dim)
        self.graph_embedder = Embedder(num_nodes, node_feats)
        self.gcn = CustomGCN(node_feats, hidden_dim)

    def forward(self, seq, node, adj, labels):
        seq_out = self.seq_mlp(seq)
        node_embd = self.graph_embedder(node)
        graph_out = self.gcn(node_embd, adj)
        graph_out = graph_out.transpose(-2, -1)

        logits = torch.matmul(seq_out, graph_out)
        outputs = (logits, )
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))

            outputs = (loss, ) + outputs

        return outputs
