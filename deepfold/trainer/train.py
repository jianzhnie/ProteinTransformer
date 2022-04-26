import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def do_compute(model, batch):
    logits = model(*batch[:-1])
    return logits, batch[-1]


def run_batch(model, optimizer, data_loader, epoch_i, desc, loss_fn):
    total_loss = 0
    logits_list = []
    ground_truth = []

    for batch in tqdm(data_loader, desc=f'{desc} Epoch {epoch_i}'):
        logits, labels = do_compute(model, batch)

        loss = loss_fn(logits, labels)
        loss = np.mean(np.sum(loss, -1))
        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        total_loss += loss.item()

        logits_list.append(F.sigmoid(logits).tolist())
        ground_truth.append(labels.tolist())

    total_loss /= len(data_loader)

    logits_list = np.concatenate(logits_list)
    ground_truth = np.concatenate(ground_truth)

    metrics = None
    return total_loss, metrics
