import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from deepfold.data.goterm_dataset import OntoDataset
from deepfold.models.go_embedder import GoEmbedder
from deepfold.utils.token_utils import save_token_embeddings

sys.path.append('../')

if __name__ == '__main__':
    embedding_dim = 256
    hidden_dim = 256
    batch_size = 64
    num_epoch = 10

    dataset = OntoDataset(
        data_dir='/Users/robin/xbiome/datasets/protein',
        obo_file='/Users/robin/xbiome/datasets/protein/go.obo')

    vocab = dataset.vocab
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             collate_fn=dataset.collate_fn,
                             shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoEmbedder(len(vocab), embedding_dim, dropout=0.1)
    model.to(device)
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    total_losses = []
    for epoch in range(num_epoch):
        total_loss = 0
        for idx, batch in enumerate(data_loader):
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            loss_term, loss_neighbor, loss_namespace = model(**batch)
            loss = loss_term + loss_neighbor + loss_namespace
            loss.backward()
            optimizer.step()
            print(
                'TermLoss: %f, NeighborLoss: %f, NamespaceLoss:%f , Loss : %f '
                % (loss_term, loss_neighbor, loss_namespace, loss))
        total_losses.append(total_loss)
        # 保存词向量（model.embeddings）
        save_token_embeddings(vocab, model.embeddings.weight.data,
                              'goterm.vec')
