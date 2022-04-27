import torch.nn as nn
from torch.nn import BCEWithLogitsLoss


class LstmEncoderModel(nn.Layer):
    """LstmEncoderModel."""
    def __init__(self,
                 vocab_size=20,
                 embed_dim=128,
                 hidden_size=1024,
                 n_layers=3,
                 num_labels=None,
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
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input):
        """forward."""
        token_embed = self.embedding(input)
        encoder_output, _ = self.lstm_encoder(token_embed)
        encoder_output = self.dropout(encoder_output)
        logits = self.classifier(encoder_output)

        return logits


class MultiLabelSequenceClassification():
    """Bert model adapted for multi-label sequence classification."""
    def __init__(self, config, pos_weight=None):
        super(self).__init__(config)
        self.num_labels = config.num_labels
        self.lstm = LstmEncoderModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pos_weight = pos_weight

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, ) + outputs[
            2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            outputs = (loss, ) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
