import torch.nn as nn


class ResnetBasicBlock(nn.Module):
    """ResnetBasicBlock."""
    def __init__(self,
                 inplanes=256,
                 planes=256,
                 kernel_size=9,
                 dilation=1,
                 dropout_rate=0.1):
        super(ResnetBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inplanes,
                               out_channels=planes,
                               kernel_size=kernel_size,
                               dilation=dilation,
                               padding='same')
        self.bn1 = nn.BatchNorm1d(planes)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv1d(in_channels=planes,
                               out_channels=planes,
                               kernel_size=kernel_size,
                               dilation=dilation,
                               padding='same')
        self.bn2 = nn.BatchNorm1d(planes)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """forward."""
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu2(out)
        out = self.dropout2(out)
        out += identity
        return out
