import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.aminoacids import MAXLEN


# Defining Model: deepgoplus
class DeepGOPlusModel(nn.Module):

    def __init__(self, nb_classes, params):
        super().__init__()

        conv_var = [
            f'conv_{i}'
            for i in range(1, np.int32(np.ceil(params['max_kernel'] / 8)))
        ]
        fc_var = [f'fc_{i}' for i in range(params['fc_depth'])]

        self.kernels = range(8, params['max_kernel'], 8)

        self.conv_ls = []
        self.pool_ls = []

        for i, kernel in enumerate(self.kernels):

            cmd1 = f"self.{conv_var[i]}=nn.Conv1d(in_channels=21, out_channels={params['nb_filters']}, kernel_size={kernel}, padding=1)"
            cmd2 = f'self.conv_ls.append(self.{conv_var[i]})'

            exec(cmd1)  # 执行语句1
            exec(cmd2)  # 执行语句2

            pool_kernel = MAXLEN - kernel + 3
            self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=1)
            self.pool_ls.append(self.pool)

        # in feature number between conv1d and fully connected layers
        nb_fc_in_features = len(self.kernels) * params['nb_filters']

        self.fc_ls = []

        for i in range(params['fc_depth']):
            cmd3 = f'self.{fc_var[i]}=nn.Linear(in_features=nb_fc_in_features, out_features={nb_classes})'
            cmd4 = f'self.fc_ls.append(self.{fc_var[i]})'

            exec(cmd3)  # 执行语句3
            exec(cmd4)  # 执行语句4

            nb_fc_in_features = nb_classes

        self.fc = nn.Linear(in_features=nb_fc_in_features,
                            out_features=nb_classes)

    def forward(self, x):

        conv_output_ls = []
        x = x.permute(0, 2, 1)
        for i, kernel in enumerate(self.kernels):

            conv_output = self.conv_ls[i](x)
            pool_output = self.pool_ls[i](conv_output)

            fla = torch.flatten(pool_output, start_dim=1)
            conv_output_ls.append(fla)
        output = torch.cat(conv_output_ls, axis=1)

        for i in range(len(self.fc_ls)):
            output = self.fc_ls[i](output)
            output = F.relu(output)

        output = self.fc(output)
        output = torch.sigmoid(output)
        return output