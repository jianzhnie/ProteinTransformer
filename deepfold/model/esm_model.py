import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Esm(nn.Module):

    def __init__(self, esm_model, nb_embedding=768, nb_classes=1000):
        super(esm_model, self).__init__()
        self.backbone, alphabet = torch.hub.load('facebookresearch/esm:main',
                                                 self.esm_model)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(nb_embedding, nb_classes)

    def forward(self, x):
        embedding = self.backbone(x, repr_layers=[5])['representations'][5]
        embedding = torch.mean(embedding, axis=1)
        x = self.fc(embedding)
        x = torch.sigmoid(x)
        return x
