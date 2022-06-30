import logging
import os
import urllib.request
from typing import Dict, List, Union

import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo
from torch.nn import BCEWithLogitsLoss

logger = logging.getLogger(__name__)


class Seq2VecEmbedder(nn.Module):
    """SeqVec Embedder. Heinzinger, Michael, et al. "Modeling aspects of the
    language of life through transfer-learning protein sequences." BMC
    bioinformatics 20.1 (2019): 723. https://doi.org/10.1186/s12859-019-3220-8.

    Compute ELMo representations using a pre-trained bidirectional language model.

    See "Deep contextualized word representations", Peters et al. for details.

    This module takes character id input and computes `num_output_representations` different layers
    of ELMo representations.  Typically `num_output_representations` is 1 or 2.  For example, in
    the case of the SRL model in the above paper, `num_output_representations=1` where ELMo was included at
    the input token representation layer.  In the case of the SQuAD model, `num_output_representations=2`
    as ELMo was also included at the GRU output layer.

    In the implementation below, we learn separate scalar weights for each output layer,
    but only run the biLM once on each input sequence for efficiency.

    # Parameters
    model_dir: `str`, required, to save  ELMo JSON options file and ELMo hdf5 weight file
    num_output_representations : `int`, required.
        The number of ELMo representation to output with
        different linear weighted combination of the 3 layers (i.e.,
        character-convnet output, 1st lstm output, 2nd lstm output).
    requires_grad : `bool`, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    do_layer_norm : `bool`, optional, (default = `False`).
        Should we apply layer normalization (passed to `ScalarMix`)?
    dropout : `float`, optional, (default = `0.5`).
        The dropout to be applied to the ELMo representations.
    """
    def __init__(self,
                 model_dir: str,
                 num_labels: int = 1000,
                 dropout_ratio: float = 0.1,
                 pool_mode: str = 'CNN',
                 num_output_representations: int = 3) -> None:
        super().__init__()
        self.elmo = self.get_elmo_model(model_dir, num_output_representations)
        self.output_dim = self.elmo.get_output_dim()
        self.pool_mode = pool_mode
        self.dropout = nn.Dropout(dropout_ratio)
        self.classifier = nn.Linear(self.output_dim, num_labels)

    def forward(
        self,
        inputs: torch.Tensor,
        word_inputs: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """get the ELMo word embedding vectors for a sentences."""
        elmo_outputs = self.elmo(inputs)
        elmo_representations = elmo_outputs['elmo_representations']
        # elmo_representations = torch.stack(elmo_representations)
        # elmo_representations = torch.transpose(elmo_representations, 0, 1)
        embeddings = self.process_embedding(elmo_representations,
                                            per_protein=True)

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
        # elmo_representations = model_outputs['elmo_representations']
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

    def process_embedding(self, embedding: List[torch.tensor],
                          per_protein: bool, layer: str) -> torch.tensor:
        """Direct output of ELMo has shape (3, L,1024), with L being the
        protein's length, 3 being the number of layers used to train SeqVec (1
        CharCNN, 2 LSTMs) and 1024 being a hyperparameter chosen to describe
        each amino acid.

        When a representation on residue level is required, you can sum over the first dimension, resulting in a tensor of size (L,1024), or just extract a
        specific layer. If you want to reduce each protein to a fixed-size vector, regardless of its length, you can average over dimension L.
        """
        #  3 * B * L * 1024
        if layer == 'sum':
            # sum over residue-embeddings of all layers (3k->1k)
            embedding = torch.stack(embedding)
            # 3 * B * L * 1024  ==>  B * 3 * L * 1024
            embedding = torch.transpose(embedding, 0, 1)
            # B * 3 * L * 1024  ==>  B * L * 1024
            embedding = torch.sum(embedding, dim=1)
        elif layer == 'CNN':
            embedding = embedding[0]
        elif layer == 'LSTM1':
            embedding = embedding[1]
        elif layer == 'LSTM2':
            embedding = embedding[2]
        else:
            # Stack the layer   3 * (B, L,1024) -> (L,3072)
            embedding = [torch.cat(embeds, dim=1) for embeds in embedding]
            embedding = torch.stack(embedding, dim=0)
        if per_protein:  # if embeddings are required on the level of whole proteins
            #  B * L * 3072/1024 ==> B * 3072/1024
            embedding = torch.mean(embedding, dim=1)
        return embedding

    def get_elmo_model(self, model_dir, num_output_representations) -> Elmo:
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
                    num_output_representations=num_output_representations)
