from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch

from deepfold.scheduler.lr_scheduler import lr_update


class LightningModule(pl.LightningModule):
    """Create lightning model to use ddp."""
    def __init__(
        self,
        model,
        lr: float,
        warmup_end_lr: float,
        warmup_updates: int = 10,
        warmup_init_lr: float = 1e-7,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.automatic_optimization = True
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = min(warmup_init_lr, lr)
        self.lr_step = (warmup_end_lr - self.warmup_init_lr) / warmup_updates
        self.decay_factor = warmup_end_lr * warmup_updates**0.5

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return output

    def configure_optimizers(
            self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            - list of optimizers.
            - list of lr schedulers.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

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

    def training_step(self, train_batch, batch_idx):
        output = self.forward(**train_batch)

        loss = output.loss

        self.log_dict(
            {'train_loss': loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        """Log the loss and metrics for a batch.

        Args:
            batch: batch input.
            batch_idx: index of the batch.
        """
        output = self.forward(**val_batch)
        loss = output.loss
        self.log_dict(
            {
                'val_loss': loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss
