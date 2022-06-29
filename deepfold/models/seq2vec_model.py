import logging
import os
import urllib.request
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo
from torch.nn import BCEWithLogitsLoss

logger = logging.getLogger(__name__)


class Seq2VecEmbedder(nn.Module):
    """SeqVec Embedder.

    Heinzinger, Michael, et al. "Modeling aspects of the language of life through transfer-learning protein sequences." BMC bioinformatics 20.1 (2019): 723.
    https://doi.org/10.1186/s12859-019-3220-8
    """
    def __init__(
        self,
        model_dir,
        num_labels: int = 1000,
        dropout_ratio: float = 0.1,
        pool_mode: str = 'mean',
    ) -> None:

        super().__init__()
        self.elmo = self.get_elmo_model(model_dir)
        self.output_dim = self.elmo.get_output_dim()
        self.pool_mode = pool_mode
        self.dropout = nn.Dropout(dropout_ratio)
        self.classifier = nn.Linear(self.output_dim, num_labels)

    def forward(
        self,
        inputs: torch.Tensor,
        word_inputs: torch.Tensor = None,
        lengths: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """get the ELMo word embedding vectors for a sentences."""
        embeddings = self.elmo(inputs)
        last_hidden_state = embeddings['elmo_representations'][0]

        if self.pool_mode == 'mean':
            embeddings = torch.mean(last_hidden_state, 1)

        if self.pool_mode == 'max':
            embeddings = torch.max(last_hidden_state, 1)

        elif self.pool_mode == 'cls':
            embeddings = last_hidden_state[:, 0]

        pooled_output = self.dropout(embeddings)
        logits = self.classifier(pooled_output)

        outputs = (logits, )

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))

            outputs = (loss, ) + outputs

        return outputs

    def compute_embeddings(
        self,
        inputs: torch.Tensor,
        word_inputs: torch.Tensor = None,
        lengths: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:

        model_outputs = self.elmo(inputs)
        elmo_representations = model_outputs['elmo_representations']
        print(elmo_representations.shape)
        last_hidden_state = model_outputs['elmo_representations'][0]
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

    def process_embedding(self, embedding: np.ndarray, per_protein: bool,
                          layer: str) -> np.ndarray:
        """Direct output of ELMo has shape (3,L,1024), with L being the
        protein's length, 3 being the number of layers used to train SeqVec (1
        CharCNN, 2 LSTMs) and 1024 being a hyperparameter chosen to describe
        each amino acid.

        When a representation on residue level is required, you can sum over the first dimension, resulting in a tensor of size (L,1024), or just extract a
        specific layer. If you want to reduce each protein to a fixed-size vector, regardless of its length, you can average over dimension L.
        """
        if layer == 'sum':
            # sum over residue-embeddings of all layers (3k->1k)
            embedding = embedding.sum(axis=0)
        elif layer == 'CNN':
            embedding = embedding[0]
        elif layer == 'LSTM1':
            embedding = embedding[1]
        elif layer == 'LSTM2':
            embedding = embedding[2]
        else:
            # Stack the layer (3,L,1024) -> (L,3072)
            embedding = np.concatenate(embedding, axis=1)
        if per_protein:  # if embeddings are required on the level of whole proteins
            embedding = embedding.mean(axis=0)
        return embedding

    def get_elmo_model(self, model_dir) -> Elmo:
        weights_path = os.path.join(model_dir, 'weights.hdf5')
        options_path = os.path.join(model_dir, 'options.json')
        # if no pre-trained model is available, yet --> download it
        if not (os.path.exists(weights_path) and os.path.exists(options_path)):
            logger.info('Model dir %s' % model_dir)
            logger.info(
                'No existing model found. Start downloading pre-trained SeqVec (~360MB)...'
            )

            os.makedirs(model_dir, exist_ok=True)
            repo_link = 'http://rostlab.org/~deepppi/embedding_repo/embedding_models/seqvec'
            options_link = repo_link + '/options.json'
            weights_link = repo_link + '/weights.hdf5'
            urllib.request.urlretrieve(options_link, str(options_path))
            urllib.request.urlretrieve(weights_link, str(weights_path))

        logger.info('Loading the model')
        return Elmo(weight_file=str(weights_path),
                    options_file=str(options_path),
                    num_output_representations=1)
