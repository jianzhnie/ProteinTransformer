import math

import numpy as np


def lr_update(
    num_updates: int,
    warmup_updates: int,
    warmup_init_lr: float,
    lr_step: float,
    decay_factor: float,
) -> float:
    """InverseSquareRootSchedule.

    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py#L32

    Args:
        num_updates: number of batches already used.
        warmup_updates: number of batch steps for warm up.
        warmup_init_lr: initial learning rate.
        lr_step: step for increasing learning rate during warm up.
        decay_factor: factor for decreasing learning rate after warm up.

    Returns:
        learning rate multiplicate factor
    """
    if num_updates < warmup_updates:
        lr = warmup_init_lr + num_updates * lr_step
    else:
        lr = decay_factor * num_updates**-0.5
    if warmup_init_lr > 0:
        return lr / warmup_init_lr

    return 0


class StepLRScheduler():
    """step learning rate scheduler."""
    def __init__(self,
                 optimizer,
                 base_lr,
                 steps,
                 decay_factor,
                 warmup_length,
                 logger=None):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.steps = steps
        self.decay_factor = decay_factor
        self.warmup_length = warmup_length

    def step(self, epoch):
        if epoch < self.warmup_length:
            lr = self.base_lr * (epoch + 1) / self.warmup_length
        else:
            lr = self.base_lr
            for s in self.steps:
                if epoch >= s:
                    lr *= self.decay_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class LinearLRScheduler():
    """Linear learning rate decay scheduler."""
    def __init__(self, optimizer, base_lr, epochs, warmup_length):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.epochs = epochs
        self.warmup_length = warmup_length

    def step(self, epoch):
        if epoch < self.warmup_length:
            lr = self.base_lr * (epoch + 1) / self.warmup_length
        else:
            e = epoch - self.warmup_length
            es = self.epochs - self.warmup_length
            lr = self.base_lr * (1 - (e / es))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class ExponentialLRScheduler():
    """Exponential learning rate decay scheduler."""
    def __init__(self,
                 optimizer,
                 base_lr,
                 epochs,
                 warmup_length=0,
                 final_multiplier=0.001,
                 decay_factor=None,
                 decay_step=1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.epochs = epochs
        self.warmup_length = warmup_length
        self.final_multiplier = final_multiplier
        self.decay_factor = decay_factor
        self.decay_step = decay_step

    def step(self, epoch):
        es = self.epochs - self.warmup_length

        if self.decay_factor is not None:
            epoch_decay = self.decay_factor
        else:
            epoch_decay = np.power(
                2,
                np.log2(self.final_multiplier) /
                math.floor(es / self.decay_step))
        if epoch < self.warmup_length:
            lr = self.base_lr * (epoch + 1) / self.warmup_length
        else:
            e = epoch - self.warmup_length
            lr = self.base_lr * (epoch_decay**math.floor(e / self.decay_step))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class CosineLRScheduler():
    """Linear learning rate decay scheduler."""
    def __init__(self, optimizer, base_lr, epochs, warmup_length, end_lr=0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.epochs = epochs
        self.warmup_length = warmup_length
        self.end_lr = end_lr

    def step(self, epoch):
        if epoch < self.warmup_length:
            lr = self.base_lr * (epoch + 1) / self.warmup_length
        else:
            e = epoch - self.warmup_length
            es = self.epochs - self.warmup_length
            lr = self.end_lr + (0.5 * (1 + np.cos(np.pi * e / es)) *
                                (self.base_lr - self.end_lr))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
