from typing import Tuple

import esm
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from deepfold.utils.constant import DEFAULT_ESM_MODEL, ESM_LIST


class Esm(nn.Module):
    def __init__(self, model_dir: str, num_labels: int = 1000):
        super().__init__()

        if model_dir not in ESM_LIST:
            print(
                f"Model dir '{model_dir}' not recognized. Using '{DEFAULT_ESM_MODEL}' as default"
            )
            model_dir = DEFAULT_ESM_MODEL

        self._model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            model_dir)
        self.num_layers = self._model.num_layers
        repr_layers = -1
        self.repr_layers = (repr_layers + self.num_layers +
                            1) % (self.num_layers + 1)
        self.hidden_size = self._model.args.embed_dim
        self.is_msa = 'msa' in model_dir

        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def _freeze_backbone(self):
        for p in self._model.parameters():
            p.requires_grad = False

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
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
                    * embeddings [num_seqs, max_len_seqs+1, embedding_size]
        """
        model_outputs = self._model(
            input_ids,
            repr_layers=[self.repr_layers],
        )
        # batch_logits = model_outputs['logits']
        batch_embeddings = model_outputs['representations'][self.repr_layers]

        pooled_output = self.dropout(batch_embeddings)
        logits = self.classifier(pooled_output)

        outputs = (logits, )

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            outputs = (loss, ) + outputs

        return outputs
