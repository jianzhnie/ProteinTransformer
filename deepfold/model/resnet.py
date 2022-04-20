import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    def __init__(self, in_channels=1024, out_channels=512):
        super(BottleNeck, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.conv(x)
        x = x.transpose(-1, -2)
        return x


class ResNetBlock(nn.Module):
    def __init__(self,
                 channels=1024,
                 kernel_size=9,
                 dilation=3,
                 cardinality=32,
                 bottle_neck_rate=0.5):
        super(ResNetBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)  # 增加bottleneck
        assert bottle_neck_rate <= 1.0, 'Unexpected vaule of bottle_neck_rate.'
        reduce = int(channels * bottle_neck_rate)
        self.bt1 = BottleNeck(channels, reduce)
        self.norm2 = nn.LayerNorm(reduce)
        self.conv_layer = nn.Conv1d(reduce,
                                    reduce,
                                    kernel_size,
                                    dilation=dilation,
                                    padding='same',
                                    groups=cardinality)
        self.bt2 = BottleNeck(reduce, channels)

    def forward(self, x):
        y = self.norm1(x)
        y = self.bt1(y)
        y = self.norm2(y)
        y = F.relu(y)
        y = y.transpose(-1, -2)
        y = self.conv_layer(y)
        y = y.transpose(-1, -2)
        y = self.bt2(y)
        y += x
        return y


class Ontology_resnet(nn.Module):
    """backbone=backbone,nb_gos, nb_zero_gos,nb_rels,channels=1024,kernel_size=
    9,dilation_rate=3,seq_len=1023,nb_residuals=5."""
    def __init__(self,
                 backbone=backbone,
                 nb_gos=5874,
                 nb_zero_gos=10000,
                 nb_rels=7,
                 channels=1024,
                 kernel_size=9,
                 dilation_rate=3,
                 seq_len=1023,
                 nb_residuals=5,
                 bottle_neck_rate=0.5,
                 cardinality=32,
                 embed_dim=1024,
                 margin=0.1):
        super(Ontology_resnet, self).__init__()
        self.nb_gos = nb_gos
        self.nb_zero_gos = nb_zero_gos

        # sequence embedding
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 初始卷积层
        self.init_conv_layer = nn.Conv1d(768,
                                         channels,
                                         kernel_size,
                                         padding='same')
        # 残差块
        blocks = []
        for i in range(
                nb_residuals
        ):  #channels=8,kernel_size=3,dilation=1,cardinality=1,bottle_neck_rate=0.5
            blocks.append(
                ResNetBlock(channels,
                            kernel_size,
                            dilation_rate**i,
                            cardinality=cardinality,
                            bottle_neck_rate=0.5))  #每层的dilation_rate不一样
        self.residual_blocks = nn.Sequential(*blocks)
        self.pool = nn.MaxPool1d(seq_len)

        # class embedding
        self.embed_dim = embed_dim
        # RuntimeError: legacy constructor expects device type: cpu but device type: cuda was passed
        self.hasFuncIndex = torch.LongTensor([nb_rels
                                              ]).to(device)  # 考虑直接在device上面创建
        self.go_embed = nn.Embedding(nb_gos + nb_zero_gos, embed_dim)
        self.go_norm = nn.BatchNorm1d(embed_dim)
        k = math.sqrt(1 / embed_dim)
        nn.init.uniform_(self.go_embed.weight, -k, k)
        self.go_rad = nn.Embedding(nb_gos + nb_zero_gos, 1)
        nn.init.uniform_(self.go_rad.weight, -k, k)

        self.rel_embed = nn.Embedding(nb_rels + 1, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, -k, k)
        self.all_gos = torch.arange(self.nb_gos, device=device)
        self.margin = margin

    def forward(self, x):
        # 输入的x形状是(batch_size,1024)
        x = self.backbone(x, repr_layers=[5])['representations'][5]
        x = x[:, 1:]  # 考虑不用直接去掉
        x = x.transpose(-1, -2)
        x = self.init_conv_layer(x)
        x = x.transpose(-1, -2)
        x = self.residual_blocks(x)
        x = x.transpose(-1, -2)
        x = self.pool(x)  #权重连接
        x = x.transpose(-1, -2)
        x_embedding = x.squeeze(-2)

        go_embed = self.go_embed(self.all_gos)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = torch.abs(self.go_rad(self.all_gos).view(1, -1))
        x_y_embedding = torch.matmul(x_embedding, hasFuncGO.T) + go_rad
        logits = torch.sigmoid(x_y_embedding)
        return logits

    def el_loss(self, go_normal_forms):
        nf1, nf2, nf3, nf4 = go_normal_forms
        nf1_loss = self.nf1_loss(nf1)
        nf2_loss = self.nf2_loss(nf2)
        nf3_loss = self.nf3_loss(nf3)
        nf4_loss = self.nf4_loss(nf4)
        # print()
        # print(nf1_loss.detach().item(),
        #       nf2_loss.detach().item(),
        #       nf3_loss.detach().item(),
        #       nf4_loss.detach().item())
        return (nf1_loss + nf3_loss + nf4_loss + nf2_loss) / self.nb_gos

    def class_dist(self, data):

        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        return dist

    def nf1_loss(self, data):
        pos_dist = self.class_dist(data)
        loss = th.mean(th.relu(pos_dist - self.margin))
        return loss

    def nf2_loss(self, data):
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        e = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        re = th.abs(self.go_rad(data[:, 2]))

        sr = rc + rd
        dst = th.linalg.norm(c - d, dim=1, keepdim=True)
        dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
        loss = th.mean(
            th.relu(dst - sr - self.margin) +
            th.relu(dst2 - rc - self.margin) +
            th.relu(dst3 - rd - self.margin))

        return loss

    def nf3_loss(self, data):
        # R some C subClassOf D
        n = data.shape[0]
        # rS = self.rel_space(data[:, 0])
        # rS = rS.reshape(-1, self.embed_dim, self.embed_dim)
        rE = self.rel_embed(data[:, 0])
        c = self.go_norm(self.go_embed(data[:, 1]))
        d = self.go_norm(self.go_embed(data[:, 2]))
        # c = th.matmul(c, rS).reshape(n, -1)
        # d = th.matmul(d, rS).reshape(n, -1)
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))

        rSomeC = c + rE
        euc = th.linalg.norm(rSomeC - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(euc + rc - rd - self.margin))
        return loss

    def nf4_loss(self, data):
        # C subClassOf R some D
        n = data.shape[0]
        c = self.go_norm(self.go_embed(data[:, 0]))
        rE = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(data[:, 2]))

        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        sr = rc + rd
        # c should intersect with d + r
        rSomeD = d + rE
        dst = th.linalg.norm(c - rSomeD, dim=1, keepdim=True)
        loss = th.mean(th.relu(dst - sr - self.margin))
        return loss
