import torch
import torch.nn as nn
import torch.nn.functional as F


class GoEmbedder(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.0) -> None:
        super().__init__()

        self.embedding = nn.Linear(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size, vocab_size)
        self.fc2 = nn.Linear(embed_size, vocab_size)
        self.fc3 = nn.Linear(embed_size, 3)
        self.activate = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.nll_loss = nn.NLLLoss()
        self.init_weights()

    def forward(self, terms, neighbors, namespace):
        """forward."""
        embed = self.embedding(terms)
        hidden = self.activate(embed)
        out1 = self.fc1(hidden)
        out2 = self.fc2(hidden)
        out3 = self.fc3(hidden)
        log_probs_term = F.log_softmax(out1, dim=1)
        log_probs_neighbor = F.log_softmax(out2, dim=1)

        loss_term = self.nll_loss(log_probs_term, terms)
        loss_neighbor = self.nll_loss(log_probs_neighbor, neighbors)
        loss_namespace = F.cross_entropy(out3, namespace)
        return loss_term, loss_neighbor, loss_namespace

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' not in name:
                torch.nn.init.uniform_(param, a=-0.1, b=0.1)
