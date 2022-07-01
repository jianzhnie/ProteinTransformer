from __future__ import division

from typing import List

import torch


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified
    values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[:k].reshape(-1).float().sum(0) * 100. / batch_size
        for k in topk
    ]


def jaccard_similarity(list1, list2):
    """Calculate the Jaccard Similarity of two lists containing strings."""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    try:
        iou = float(intersection) / union
    except ZeroDivisionError:
        iou = 0
    return iou


def compute_jaccard_matrix(list1: List[List[str]], list2: List[List[str]]):
    M, N = len(list1), len(list2)
    jaccardMat = [[0] * N] * M
    for i in range(M):
        l1 = list1[i]
        for j in range(N):
            l2 = list2[j]
            jaccardMat[i][j] = jaccard_similarity(l1, l2)
    return jaccardMat


if __name__ == '__main__':
    list1 = [['a', 'b', 'c'], ['a', 'b', 'd'], ['a', 'b', 'c']]
    list2 = [['e', 'b', 'c'], ['a', 'f', 'd']]

    mat = compute_jaccard_matrix(list1, list2)
    print(mat)
