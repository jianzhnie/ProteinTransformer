from __future__ import division

from typing import List


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
