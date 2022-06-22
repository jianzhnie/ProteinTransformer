import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2vecLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # get log probabilities only for neighbors
        neighbors_probs = torch.masked_select(y_pred, y_true)
        loss = torch.mean(torch.log(neighbors_probs))
        return -loss


class GoEmbedder(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.0) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size, vocab_size)
        self.fc2 = nn.Linear(embed_size, vocab_size)
        self.fc3 = nn.Linear(embed_size, 3)
        self.activate = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.nll_loss = nn.NLLLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.init_weights()

    def forward_(self, term_ids, neighbor_ids, labels, retrun_dict=True):
        """forward."""
        embed = self.embedding(term_ids)
        hidden = self.activate(embed)
        out1 = self.fc1(hidden)
        out2 = self.fc2(hidden)
        out3 = self.fc3(hidden)
        # term
        log_probs_term = F.log_softmax(out1, dim=1)
        # neighbor
        log_probs_neighbor = F.log_softmax(out2, dim=1)

        loss_term = self.nll_loss(log_probs_term, term_ids)
        loss_neighbor = self.nll_loss(log_probs_neighbor, neighbor_ids)

        # three sub module
        loss_namespace = F.cross_entropy(out3, labels)
        return loss_term, loss_neighbor, loss_namespace

    def forward(self, term_ids, neighbor_ids, labels, retrun_dict=True):
        """forward."""
        embed = self.embedding(term_ids)
        hidden = self.activate(embed)
        out1 = self.fc1(hidden)
        out2 = self.fc2(hidden)
        out3 = self.fc3(hidden)
        # term
        log_probs_term = F.log_softmax(out1, dim=1)
        # neighbor
        log_probs_neighbor = F.log_softmax(out2, dim=1)

        # loss
        loss_term = self.nll_loss(log_probs_term, term_ids)
        loss_neighbor = self.nll_loss(log_probs_neighbor, neighbor_ids)
        # three sub module
        loss_namespace = F.cross_entropy(out3, labels)
        return loss_term, loss_neighbor, loss_namespace

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' not in name:
                torch.nn.init.uniform_(param, a=-0.1, b=0.1)
