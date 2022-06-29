import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# GCN module
class GraphConvolution(nn.Module):
    def __init__(self, nfeat, nhid, bias=True):
        super(GraphConvolution, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.weight = nn.Parameter(torch.FloatTensor(nfeat, nhid))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(nhid))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = torch.mm(input, self.weight)
        output = torch.spmm(adj, x)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        return x


# attention  module
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项 Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作."""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """缩放点积注意力."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状."""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力."""
    def __init__(self,
                 key_size,
                 query_size,
                 value_size,
                 num_hiddens,
                 num_heads,
                 dropout,
                 bias=False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class P2GO(nn.Module):
    def __init__(self,
                 backbone,
                 aa_dim=1280,
                 latent_dim=128,
                 dropout_rate=0.1,
                 n_head=2,
                 batch_size=4,
                 nb_classes=5874):
        super().__init__()
        # backbone
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        # AA_emebdding transform
        self.aa_transform = nn.Linear(aa_dim, latent_dim)

        # go embedder
        self.go_embedder = nn.Sequential(
            nn.Embedding(self.nb_classes, latent_dim), nn.ReLU(inplace=True))
        self.fc1 = nn.Linear(latent_dim, self.nb_classes)
        self.fc2 = nn.Linear(latent_dim, self.nb_classes)
        self.fc3 = nn.Linear(latent_dim, 3)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.nll_loss = nn.NLLLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

        # label-wise attention
        self.attention = MultiHeadAttention(latent_dim,
                                            latent_dim,
                                            latent_dim,
                                            latent_dim,
                                            num_heads=n_head,
                                            dropout=dropout_rate)
        # weight alpha
        self.alpha = torch.nn.Parameter(torch.ones((1, nb_classes)) * 10000.0,
                                        requires_grad=True)

        # gnn module
        self.gcn = GCN(latent_dim, latent_dim)
        # output layer
        self.go_transform2 = nn.Sequential(
            nn.Linear(int(2 * latent_dim), latent_dim), nn.ReLU())

    def forward(self, x, valid_len, adj, term_ids):
        # backbone
        x = self.backbone(x, repr_layers=[33])['representations'][33]
        x = x[:, 1:]
        # x [B,L,C]
        # AA_embedding transform
        x = self.aa_transform(x)
        mean_list = []
        for i, length in enumerate(list(valid_len)):
            length = int(length)
            mean = x[i, :length].mean(axis=0, keepdim=True)
            mean_list.append(mean)
        mean_batch = torch.cat(mean_list)
        mean_embedding = mean_batch.unsqueeze(1)
        mean_embedding = mean_embedding.repeat((1, self.nb_classes, 1))
        # go embedder
        # go_embedding [nb_classes,latent_dim]
        go_embedding = self.go_embedder(term_ids)
        # label-wise attention
        y_embedding = go_embedding.repeat((x.shape[0], 1, 1))
        label_attn_embedding = self.attention(y_embedding, x, x, valid_len)
        # feature combination
        alpha = torch.sigmoid(self.alpha)
        alpha = alpha.unsqueeze(-1)
        alpha = alpha.repeat((x.shape[0], 1, label_attn_embedding.shape[2]))
        label_level_embedding = alpha * label_attn_embedding + (
            1 - alpha) * mean_embedding
        # gnn module
        go_out = self.gcn(go_embedding, adj)
        # output layer
        go_out = torch.cat((go_embedding, go_out), dim=1)
        go_out = self.go_transform2(go_out)
        go_out = go_out.repeat((x.shape[0], 1, 1))
        out = torch.sum(go_out * label_level_embedding, dim=-1)

        return torch.sigmoid(out)

    def go_loss_fn(self, term_ids, ancesstors, sub_ontology):
        return self.term_loss_fn(term_ids) + self.anc_loss_fn(
            term_ids, ancesstors) + self.sub_ont_loss_fn(
                term_ids, sub_ontology)

    def term_loss_fn(self, term_ids):
        embeddings = self.go_embedder(term_ids)
        out1 = self.fc1(embeddings)
        log_probs_term = F.log_softmax(out1, dim=1)
        loss_term = self.nll_loss(log_probs_term, term_ids)
        return loss_term

    def anc_loss_fn(self, term_ids, neighbor_ids):
        embeddings = self.go_embedder(term_ids)
        out2 = self.fc2(embeddings)
        log_probs_neighbor = F.log_softmax(out2, dim=1)
        loss_neighbor = self.nll_loss(log_probs_neighbor, neighbor_ids)
        return loss_neighbor

    def sub_ont_loss_fn(self, term_ids, sub_ontology):
        embeddings = self.go_embedder(term_ids)
        out3 = self.fc3(embeddings)
        loss_namespace = F.cross_entropy(out3, sub_ontology)
        return loss_namespace


if __name__ == '__main__':
    # data, target, valid_len = next(iter(test_dataloader))
    # data = data.squeeze(1)
    # target = target.squeeze(1)
    # device = 'cuda:0'
    # adj, go_embedding, data, target, valid_len = adj.to(device), go_embedding.to(device), data.to(device), target.to(device), valid_len.to(device)
    # model = P2GO(backbone=backbone, go_dim=params['go_dim'],aa_dim=params['aa_dim'],
    #                         latent_dim=params['latent_dim'],dropout_rate=params['dropout_rate'],
    #                         n_head=params['n_head'],batch_size=params['train_batch_size'],nb_classes=nb_classes)
    # model = model.to(device)
    # out = model(data, valid_len, adj, go_embedding)

    # nn.NLLLoss()
    device = 'cuda:0'
    embedding = nn.Embedding(5874, 512)
    fc1 = nn.Linear(512, 5874)
    fc2 = nn.Linear(512, 5874)
    fc3 = nn.Linear(512, 3)

    ids = torch.arange(5874)
    go_embedding = embedding(ids)

    embedding, fc1, fc2, fc3, ids = embedding.to(device), fc1.to(
        device), fc2.to(device), fc3.to(device), ids.to(device)
    go_embedding = go_embedding.to(device)
    print(go_embedding.shape)
    out1 = fc1(go_embedding)
    out2 = fc2(go_embedding)
    out3 = fc3(go_embedding)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
