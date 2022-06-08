from typing import Dict, List, Tuple, Union

import esm
import torch
import torch.nn as nn

from deepfold.utils.constant import (DEFAULT_ESM_MODEL, ESM_LIST,
                                     POOLING_MODE_LIST)

from .layers.transformer_represention import (AttentionPooling, CNNPooler,
                                              LSTMPooling,
                                              WeightedLayerPooling)


class EsmEmbeddingModel(nn.Module):
    def __init__(self, input_size=1280, num_labels=10000, dropout_ratio=0.1):
        super().__init__()

        self.hidden_size = input_size * 2
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.norm = nn.BatchNorm1d(self.hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_ratio)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, embeddings, labels=None):
        out = self.fc1(embeddings)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits


class ESMPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class EsmTransformer(nn.Module):
    """ESMTransformer."""
    def __init__(self,
                 model_dir: str = 'esm1b_t33_650M_UR50S',
                 num_labels: int = 1000,
                 max_len: int = 1024,
                 repr_layers: List[int] = None,
                 dropout_ratio: float = 0.0,
                 pool_mode: str = 'cls',
                 fintune: bool = True):
        super().__init__()

        if model_dir not in ESM_LIST:
            print(
                f"Model dir '{model_dir}' not recognized. Using '{DEFAULT_ESM_MODEL}' as default"
            )
            model_dir = DEFAULT_ESM_MODEL

        self._model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            model_dir)
        self.num_layers = self._model.num_layers
        # esm1b: 33 layers
        if repr_layers is not None:
            assert all(-(self.num_layers + 1) <= i <= self.num_layers
                       for i in repr_layers)
            self.repr_layers = [
                (i + self.num_layers + 1) % (self.num_layers + 1)
                for i in repr_layers
            ]
        else:
            self.repr_layers = list(range(self.num_layers + 1))

        self.hidden_size = self._model.args.embed_dim
        self.is_msa = 'msa' in model_dir

        self.num_labels = num_labels
        self.max_len = max_len
        self.pool_mode = pool_mode

        assert pool_mode in POOLING_MODE_LIST, (
            f"Pooling Mode '{pool_mode}' not recognized. allowed pooling method {POOLING_MODE_LIST}"
        )

        print(f"Using '{pool_mode}' Method to embedding Protein seqence")
        if pool_mode == 'pooler':
            self.pooler = ESMPooler(self.hidden_size)

        if self.pool_mode == 'cnn':
            self.pooler = CNNPooler(self.hidden_size)

        if pool_mode == 'weighted':
            self.pooler = WeightedLayerPooling(num_layers=self.num_layers,
                                               layer_start=1,
                                               layer_weights=None)

        if pool_mode == 'attention':
            self.pooler = AttentionPooling(num_layers=self.num_layers,
                                           hidden_size=self.hidden_size,
                                           hiddendim_fc=128)
        if pool_mode == 'lstm':
            self.pooler = LSTMPooling(num_layers=self.num_layers,
                                      hidden_size=self.hidden_size,
                                      hiddendim_lstm=256)

        self.dropout = nn.Dropout(dropout_ratio)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        if self.pool_mode == 'mean_max':
            self.classifier = nn.Linear(self.hidden_size * 2, num_labels)

        self.fintune = fintune
        if not self.fintune:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self._model.parameters():
            p.requires_grad = False

    def forward(self,
                input_ids,
                lengths=None,
                labels=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function which computes logits and embeddings based on a list of
        sequences, a provided batch size and an inference configuration. The
        output is obtained by computing a forward pass through the model
        ("forward inference")

        The datagenerator is not the same the multi_gpus inference. We use a tqdm progress bar
        that is updated by the worker. The progress bar is instantiated before ray.remote

        Args:
            inputs (Dict[str, torch.tensor]): [description]
        Returns:
            Tuple[torch.tensor, torch.tensor]:
                    * logits [num_seqs, max_len_seqs, vocab_size]
                    * embeddings [num_seqs, max_len_seqs, embedding_size]
        """
        model_outputs = self._model(
            input_ids,
            repr_layers=self.repr_layers,
        )
        all_hidden_states = torch.stack(
            [model_outputs['representations'][i] for i in self.repr_layers])
        last_hidden_state = model_outputs['representations'][
            self.repr_layers[-1]]
        # batch_embeddings: batch_size * seq_length * embedding_dim
        if self.pool_mode == 'mean':
            embeddings = torch.mean(last_hidden_state, 1)

        if self.pool_mode == 'max':
            embeddings = torch.max(last_hidden_state, 1)

        elif self.pool_mode == 'cls':
            embeddings = last_hidden_state[:, 0]

        elif self.pool_mode == 'mean_max':
            max_pooling_embeddings = torch.max(last_hidden_state, 1)
            mean_pooling_embeddings = torch.mean(last_hidden_state, 1)
            embeddings = torch.cat(
                [mean_pooling_embeddings, max_pooling_embeddings], 1)

        elif self.pool_mode == 'pooler':
            embeddings = self.pooler(last_hidden_state)

        elif self.pool_mode == 'cnn':
            embeddings = self.pooler(last_hidden_state)

        elif self.pool_mode == 'weighted':
            weighted_pooling_embeddings = self.pooler(all_hidden_states)
            embeddings = weighted_pooling_embeddings[:, 0]

        elif self.pool_mode == 'attention':
            embeddings = self.pooler(all_hidden_states)

        elif self.pool_mode == 'lstm':
            embeddings = self.pooler(all_hidden_states)

        pooled_output = self.dropout(embeddings)
        logits = self.classifier(pooled_output)
        return logits

    def compute_embeddings(
            self, input_ids, lengths,
            labels) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:

        model_outputs = self._model(
            input_ids,
            repr_layers=self.repr_layers,
        )
        last_hidden_state = model_outputs['representations'][
            self.repr_layers[-1]]
        # batch_embeddings: batch_size * seq_length * embedding_dim
        seqence_embeddings_list = [emb for emb in last_hidden_state]
        # Remove class token and padding
        seq_len_list = [ll for ll in lengths]
        filtered_embeddings = [
            emb[1:(length + 1), :]
            for emb, length in zip(seqence_embeddings_list, seq_len_list)
        ]
        embeddings_dict = {}
        if 'mean' in self.pool_mode:
            embeddings_dict['mean'] = torch.stack(
                [torch.mean(emb, dim=0) for emb in filtered_embeddings])
        # keep class token only
        if 'cls' in self.pool_mode:
            embeddings_dict['cls'] = torch.stack(
                [emb[0, :] for emb in seqence_embeddings_list])
        return embeddings_dict
