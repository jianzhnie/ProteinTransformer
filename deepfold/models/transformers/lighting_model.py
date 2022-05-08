import logging
from collections import OrderedDict
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)


class LightningModule(pl.LightningModule):
    """Create lightning model to use ddp."""
    def __init__(
        self,
        model=None,
        model_name_or_path: str = None,
        num_labels: int = 10,
        task_name: str = 'multi_label_classification',
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 1000,
        weight_decay: float = 1e-4,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        if model is not None:
            assert isinstance(
                model, nn.Module
            ), f'given custom network {type(model)}, torch.nn.Module expected'
        if model is not None and isinstance(model, nn.Module):
            self.model = model
        else:
            self.config = AutoConfig.from_pretrained(model_name_or_path,
                                                     num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, config=self.config)

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs

    def unfreeze_encoder(self) -> None:
        """un-freezes the encoder layer."""
        if self._frozen:
            logging.info('\n-- Encoder model fine-tuning')
            for param in self.model.bert.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """freezes the encoder layer."""
        for param in self.model.bert.parameters():
            param.requires_grad = False
        self._frozen = True

    def forward(self, **inputs):
        """Usual pytorch forward function."""
        return self.model(**inputs)

    def predict(self, **inputs) -> dict:
        """Predict function.

        :param sample: dictionary with the text we want to classify.
        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            outputs = self.forward(**inputs)
            logits = outputs[1].numpy()
            if self.hparams.task_name == 'multi_label_classification':
                preds = torch.sigmoid(logits)
            else:
                if self.hparams.num_labels > 1:
                    preds = torch.argmax(logits, dim=1)
                elif self.hparams.num_labels == 1:
                    preds = logits.squeeze()
        results = OrderedDict({'predictions': preds})
        return results

    def training_step(self, batch: tuple, batch_idx: int, **kwargs) -> dict:
        """Runs one training step. This usually consists in the forward
        function followed by the loss function.

        :param batch: The output of your dataloader.
        :param batch_idx: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        outputs = self.forward(**batch)
        train_loss = outputs[0]
        tqdm_dict = {'train_loss': train_loss}
        output = OrderedDict({
            'loss': train_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_idx: int, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """

        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        output = OrderedDict({'loss': val_loss})
        return output

    def test_step(self, batch: tuple, batch_idx: int, **kwargs) -> dict:
        outputs = self.forward(**batch)
        test_loss, logits = outputs[:2]
        output = OrderedDict({'test_loss': test_loss})
        return output

    def validation_epoch_end(self, outputs: list) -> dict:
        """Function that takes as input a list of dictionaries returned by the
        validation_step function and measures the model performance accross the
        entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """

        val_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        # preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        # labels = torch.cat([x['labels']
        #                     for x in outputs]).detach().cpu().numpy()
        tqdm_dict = {'val_loss': val_loss_mean}
        result = {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'val_loss': val_loss_mean,
        }
        return result

    def test_epoch_end(self, outputs: list) -> dict:
        """Function that takes as input a list of dictionaries returned by the
        validation_step function and measures the model performance accross the
        entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        tqdm_dict = {'test_loss': test_loss_mean}
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

    def configure_optimizers(
            self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            - list of optimizers.
            - list of lr schedulers.
        """
        parameters = [
            {
                'params': self.model.classifier.parameters()
            },
            {
                'params': self.model.bert.parameters(),
                'lr': self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = torch.optim.Adam(parameters,
                                     lr=self.hparams.learning_rate,
                                     eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [lr_scheduler]
