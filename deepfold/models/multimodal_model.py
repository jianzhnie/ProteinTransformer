import collections

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


class ProtGCNModel(nn.Module):
    """ ProtGCNModel: Kyudam Choi, Yurim Lee, Cheongwon Kim, Minsung Yoon \
        An Effective GCN-based Hierarchical Multi-label classification for Protein Function Prediction \
        arxiv, http://arxiv.org/abs/2112.02810
    """
    def __init__(self,
                 nodes: torch.Tensor,
                 adjmat: torch.tensor,
                 seq_dim: int = 1024,
                 node_feats: int = 512,
                 hidden_dim: int = 512,
                 dropout=0.1):
        super().__init__()
        assert nodes.shape[0] == adjmat.shape[0]
        self.nodesMat = nodes
        self.adjMat = adjmat
        self.num_nodes = nodes.shape[0]
        self.seq_mlp = MLP(seq_dim, hidden_dim)
        self.graph_embedder = Embedder(self.num_nodes, node_feats)
        self.gcn = CustomGCN(node_feats, hidden_dim, dropout=dropout)
        self.num_labels = self.num_nodes

    def forward(self, embeddings, labels):
        seq_out = self.seq_mlp(embeddings)
        node_embd = self.graph_embedder(self.nodesMat)
        graph_out = self.gcn(node_embd, self.adjMat)
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


class ProtPubMedBert(nn.Module):
    """ProtPubMedBert."""
    def __init__(self,
                 seq_dim=1024,
                 hidden_dim=512,
                 vector_dim=800,
                 description_dim=768,
                 emb_dim=768):

        super(ProtPubMedBert, self).__init__()
        self.fc = nn.Linear(seq_dim, hidden_dim)
        self.fc_description = nn.Linear(description_dim, hidden_dim)
        self.fc_vector = nn.Linear(vector_dim, hidden_dim)
        self.fc_embed = nn.Linear(vector_dim, emb_dim)
        self.activation = torch.nn.Sigmoid()

    def forward(self,
                seq=None,
                prot_description=None,
                prot_structure=None,
                gotext_embde=None):

        features = collections.OrderedDict()
        if 'seqs' in self.feature:
            features['seqs'] = self.fc(seq)

        if 'protein description' in self.feature:
            features['protein description'] = self.fc_description(
                prot_description)

        if 'network' in self.feature:
            features['network'] = self.fc_vector(prot_structure)

        for i in range(len(self.feature)):
            if i == 0:
                prot_cat = features[self.feature[0]]
            else:
                prot_cat = torch.cat((prot_cat, features[self.feature[i]]),
                                     dim=1)

        prot_feature = self.fc_embed(prot_cat)
        emb2 = gotext_embde.permute(1, 0)
        logits = torch.mm(prot_feature, emb2)
        probs = self.activation(logits)
        return probs
