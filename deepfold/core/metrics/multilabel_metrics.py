import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    probs = torch.sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    f1_micro_average = f1_score(y_true=labels, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(labels, y_pred, average='micro')
    accuracy = accuracy_score(labels, y_pred)
    # return as dictionary
    metrics = {
        'f1': f1_micro_average,
        'roc_auc': roc_auc,
        'accuracy': accuracy
    }
    return metrics
