import torch.nn as nn


class LstmEncoderModel(nn.Layer):
    """LstmEncoderModel."""
    def __init__(self,
                 vocab_size,
                 embed_dim=128,
                 hidden_size=1024,
                 n_layers=3,
                 bidirectional=False,
                 padding_idx=0,
                 dropout_rate=0.1):
        """__init__"""
        super(LstmEncoderModel, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size,
                                      embed_dim,
                                      padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm_encoder = nn.LSTM(embed_dim,
                                    hidden_size,
                                    num_layers=n_layers,
                                    bidirectional=bidirectional)

    def forward(self, input):
        """forward."""
        token_embed = self.embedding(input)
        encoder_output, _ = self.lstm_encoder(token_embed)
        encoder_output = self.dropout(encoder_output)

        return encoder_output
