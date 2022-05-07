import logging
from collections import OrderedDict
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import BertModel

from deepfold.scheduler.lr_scheduler import lr_update


class LightningModule(pl.LightningModule):
    """Create lightning model to use ddp."""
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.num_classes = hparams.num_classes
        self.batch_size = self.hparams.batch_size
        self.model_name = 'Rostlab/prot_bert_bfd'
        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs

    def __build_model(self) -> None:
        """Init BERT model + classification head."""
        self.ProtBertBFD = BertModel.from_pretrained(
            self.model_name,
            gradient_checkpointing=self.hparams.gradient_checkpointing)
        self.encoder_features = 1024

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features * 4, self.num_classes))

    def __build_loss(self):
        """Initializes the loss function/s."""
        self._loss = nn.BCEWithLogitsLoss()

    def unfreeze_encoder(self) -> None:
        """un-freezes the encoder layer."""
        if self._frozen:
            logging.info('Encoder model fine-tuning')
            for param in self.ProtBertBFD.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """freezes the encoder layer."""
        logging.info('Encoder model freezed')
        for param in self.ProtBertBFD.parameters():
            param.requires_grad = False
        self._frozen = True

    # https://github.com/UKPLab/sentence-transformers/blob/eb39d0199508149b9d32c1677ee9953a84757ae4/sentence_transformers/models/Pooling.py
    def pool_strategy(self,
                      features,
                      pool_cls=True,
                      pool_max=True,
                      pool_mean=True,
                      pool_mean_sqrt=True):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        # Pooling strategy
        output_vectors = []
        if pool_cls:
            output_vectors.append(cls_token)
        if pool_max:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9
            # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded,
                                       1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(
                    sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        return output_vector

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output, pooler_output=pooled_output,
        past_key_values=encoder_outputs.past_key_values,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions, )"""
        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_mask, device=self.device)
        word_embeddings = self.ProtBertBFD(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        pooling = self.pool_strategy({
            'token_embeddings':
            word_embeddings,
            'cls_token_embeddings':
            word_embeddings[:, 0],
            'attention_mask':
            attention_mask,
        })

        return {'logits': self.classification_head(pooling)}

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """Computes Loss value according to a loss function.

        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions['logits'], targets['labels'])

    def configure_optimizers(
            self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            - list of optimizers.
            - list of lr schedulers.
        """
        parameters = [
            {
                'params': self.classification_head.parameters()
            },
            {
                'params': self.ProtBertBFD.parameters(),
                'lr': self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = torch.optim.Adam(parameters, lr=self.hparams.learning_rate)
        lr_scheduler = {
            'scheduler':
            torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda x: lr_update(
                    num_updates=x,
                    warmup_updates=self.warmup_updates,
                    warmup_init_lr=self.warmup_init_lr,
                    lr_step=self.lr_step,
                    decay_factor=self.decay_factor,
                ),
            ),
            'name':
            'learning_rate',
            'interval':
            'step',
            'frequency':
            1,
        }

        return [optimizer], [lr_scheduler]

    def training_step(self, batch: tuple, batch_idx: int, *args,
                      **kwargs) -> dict:
        """Runs one training step. This usually consists in the forward
        function followed by the loss function.

        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_idx: int, *args,
                        **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch

        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        y = targets['labels']
        y_hat = model_out['logits']

        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = self.metric_acc(labels_hat, y)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })

        return output

    def validation_epoch_end(self, outputs: list) -> dict:
        """Function that takes as input a list of dictionaries returned by the
        validation_step function and measures the model performance accross the
        entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """

        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()

        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'val_loss': val_loss_mean,
        }
        return result

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_test = self.loss(model_out, targets)

        y = targets['labels']
        y_hat = model_out['logits']

        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = self.metric_acc(labels_hat, y)

        output = OrderedDict({
            'test_loss': loss_test,
            'test_acc': test_acc,
        })

        return output

    def test_epoch_end(self, outputs: list) -> dict:
        """Function that takes as input a list of dictionaries returned by the
        validation_step function and measures the model performance accross the
        entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc_mean = torch.stack([x['test_acc'] for x in outputs]).mean()

        tqdm_dict = {'test_loss': test_loss_mean, 'test_acc': test_acc_mean}
        result = {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'test_loss': test_loss_mean,
        }
        return result

    def on_epoch_end(self):
        """Pytorch lightning hook."""
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()
