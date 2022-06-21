import math

import torch


def onehot_by_for(input_tensor: torch.Tensor,
                  num_classes: int,
                  on_value: float = 1.0):
    """for循环, 对一个空白（全零）张量中的指定位置进行赋值（赋 1）操作。"""
    assert num_classes > input_tensor.max(), torch.unique(input_tensor)
    one_hot = input_tensor.new_zeros(size=(num_classes, *input_tensor.shape))
    for i in range(num_classes):
        one_hot[i, input_tensor == i] = on_value
    one_hot = one_hot.permute(1, 0)
    return one_hot


def onehot_by_scatter(input_tensor: torch.Tensor,
                      num_classes: int,
                      on_value: float = 1.0):
    """因为one-hot本身形式上的含义就是对于第i类数据，第i个位置为 1，其余位置为 0。 所以对全零 tensor
    使用scatter_是可以非常容易的构造出one-hottensor 的，即对对应于类别编号的位置放置 1 即可。"""
    assert num_classes > input_tensor.max(), torch.unique(input_tensor)
    one_hot = torch.zeros(size=(math.prod(input_tensor.shape), num_classes))
    one_hot.scatter_(dim=1, index=input_tensor.reshape(-1, 1), value=on_value)
    one_hot = one_hot.reshape(*input_tensor.shape, num_classes)
    return one_hot


def onehot_by_index_select(input_tensor: torch.Tensor, num_classes: int):
    """torch.index_select(input, dim, index, *, out=None) → Tensor.

        - input (Tensor) – the input tensor.
        - dim (int) – the dimension in which we index
        - index (IntTensor or LongTensor) – the 1-D tensor containing the indices to index

    对于原始从小到大排布的类别序号对应的one-hot编码成的矩阵就是一个单位矩阵。所以每个类别对应的就是该单位矩阵的特定的列（或者行）。

    这一需求恰好符合index_select的功能。所以我们可以使用其实现one_hot编码，只需要使用类别序号索引特定的列或者行即可
    """
    assert num_classes > input_tensor.max(), torch.unique(input_tensor)
    one_hot = torch.eye(num_classes).index_select(
        dim=0, index=input_tensor.reshape(-1))
    one_hot = one_hot.reshape(*input_tensor.shape, num_classes)
    return one_hot
