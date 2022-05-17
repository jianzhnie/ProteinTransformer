import typing

import torch
import torch.functional as F
import torch.nn as nn
from tape.models.modeling_utils import ProteinModel
from torch.nn import BCEWithLogitsLoss

URL_PREFIX = 'https://s3.amazonaws.com/songlabdata/proteindata/pytorch-models/'
LSTM_PRETRAINED_CONFIG_ARCHIVE_MAP: typing.Dict[str, str] = {}
LSTM_PRETRAINED_MODEL_ARCHIVE_MAP: typing.Dict[str, str] = {}


class ProteinLSTMConfig():
    def __init__(self,
                 vocab_size: int = 30,
                 embed_dim: int = 128,
                 hidden_size: int = 1024,
                 num_hidden_layers: int = 3,
                 bidirectional: bool = False,
                 hidden_dropout_prob: float = 0.1,
                 dropout_rate: float = 0.1,
                 initializer_range: float = 0.02,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.bidirectional = bidirectional
        self.hidden_dropout_prob = hidden_dropout_prob
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range


class ProteinLSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, inputs):
        inputs = self.dropout(inputs)
        self.lstm.flatten_parameters()
        return self.lstm(inputs)


class ProteinLSTMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scalar_reweighting = nn.Linear(2 * config.num_hidden_layers, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.scalar_reweighting(hidden_states).squeeze(2)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ProteinLSTMEncoder(nn.Module):
    def __init__(self, config: ProteinLSTMConfig):
        super().__init__()
        forward_lstm = [
            ProteinLSTMLayer(config.input_size, config.hidden_size)
        ]
        reverse_lstm = [
            ProteinLSTMLayer(config.input_size, config.hidden_size)
        ]
        for _ in range(config.num_hidden_layers - 1):
            forward_lstm.append(
                ProteinLSTMLayer(config.hidden_size, config.hidden_size,
                                 config.hidden_dropout_prob))
            reverse_lstm.append(
                ProteinLSTMLayer(config.hidden_size, config.hidden_size,
                                 config.hidden_dropout_prob))
        self.forward_lstm = nn.ModuleList(forward_lstm)
        self.reverse_lstm = nn.ModuleList(reverse_lstm)
        self.output_hidden_states = config.output_hidden_states

    def forward(self, inputs, input_mask=None):
        all_forward_pooled = ()
        all_reverse_pooled = ()
        all_hidden_states = (inputs, )
        forward_output = inputs
        for layer in self.forward_lstm:
            forward_output, forward_pooled = layer(forward_output)
            all_forward_pooled = all_forward_pooled + (forward_pooled[0], )
            all_hidden_states = all_hidden_states + (forward_output, )

        reversed_sequence = self.reverse_sequence(inputs, input_mask)
        reverse_output = reversed_sequence
        for layer in self.reverse_lstm:
            reverse_output, reverse_pooled = layer(reverse_output)
            all_reverse_pooled = all_reverse_pooled + (reverse_pooled[0], )
            all_hidden_states = all_hidden_states + (reverse_output, )
        reverse_output = self.reverse_sequence(reverse_output, input_mask)

        output = torch.cat((forward_output, reverse_output), dim=2)
        pooled = all_forward_pooled + all_reverse_pooled
        pooled = torch.stack(pooled, 3).squeeze(0)
        outputs = (output, pooled)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states, )

        return outputs  # sequence_embedding, pooled_embedding, (hidden_states)

    def reverse_sequence(self, sequence, input_mask):
        if input_mask is None:
            idx = torch.arange(sequence.size(1) - 1, -1, -1)
            reversed_sequence = sequence.index_select(1,
                                                      idx,
                                                      device=sequence.device)
        else:
            sequence_lengths = input_mask.sum(1)
            reversed_sequence = []
            for seq, seqlen in zip(sequence, sequence_lengths):
                idx = torch.arange(seqlen - 1, -1, -1, device=seq.device)
                seq = seq.index_select(0, idx)
                seq = F.pad(seq, [0, 0, 0, sequence.size(1) - seqlen])
                reversed_sequence.append(seq)
            reversed_sequence = torch.stack(reversed_sequence, 0)
        return reversed_sequence


class ProteinLSTMAbstractModel(ProteinModel):

    config_class = ProteinLSTMConfig
    pretrained_model_archive_map = LSTM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'lstm'

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class ProteinLSTMModel(ProteinLSTMAbstractModel):
    def __init__(self, config: ProteinLSTMConfig):
        super().__init__(config)
        self.embed_matrix = nn.Embedding(config.vocab_size, config.input_size)
        self.encoder = ProteinLSTMEncoder(config)
        self.pooler = ProteinLSTMPooler(config)
        self.output_hidden_states = config.output_hidden_states
        self.init_weights()

    def forward(self, input_ids, input_mask=None):
        if input_mask is None:
            input_mask = torch.ones_like(input_ids)

        # fp16 compatibility
        embedding_output = self.embed_matrix(input_ids)
        outputs = self.encoder(embedding_output, input_mask=input_mask)
        sequence_output = outputs[0]
        pooled_outputs = self.pooler(outputs[1])

        outputs = (sequence_output, pooled_outputs) + outputs[2:]
        return outputs  # sequence_output, pooled_output, (hidden_states)


class LstmEncoderModel(nn.Module):
    """LstmEncoderModel."""
    def __init__(self,
                 vocab_size=20,
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


class PretrainTaskModel(nn.Layer):
    """PretrainTaskModel."""
    def __init__(self, class_num, model_config, encoder_model):
        """__init__"""
        super(PretrainTaskModel, self).__init__()

        model_type = model_config.get('model_type', 'transformer')
        hidden_size = model_config.get('hidden_size', 512)
        in_channels = hidden_size * 2 if model_type == 'lstm' else hidden_size

        self.conv_decoder = nn.Sequential(
            nn.Conv1D(in_channels=in_channels,
                      out_channels=128,
                      kernel_size=5,
                      padding='same',
                      data_format='NLC'),
            nn.ReLU(),
            nn.Conv1D(in_channels=128,
                      out_channels=class_num,
                      kernel_size=3,
                      padding='same',
                      data_format='NLC'),
        )
        self.encoder_model = encoder_model

    def forward(self, input, pos):
        """forward."""
        encoder_output = self.encoder_model(input, pos)
        decoder_output = self.conv_decoder(encoder_output)

        return decoder_output


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
