import time

import numpy as np
import torch

from deepfold.utils.metrics import AverageMeter
from deepfold.utils.model import save_checkpoint


def train(model,
          train_loader,
          optimizer,
          lr_scheduler,
          gradient_accumulation_steps,
          epoch,
          device,
          logger,
          log_interval=1):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.train()
    steps_per_epoch = len(train_loader)
    end = time.time()
    for step, batch in enumerate(train_loader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass for multilabel classification
        outputs = model(**batch)
        data_time = time.time() - end

        loss = outputs[0]
        loss = loss / gradient_accumulation_steps
        batch_size = loss.shape[0]
        if step % gradient_accumulation_steps == 0 or step == steps_per_epoch - 1:
            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        it_time = time.time() - end
        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), batch_size)

        end = time.time()
        if (step % log_interval == 0) or (step == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                log_name = 'Train-log'
                logger.info(
                    '{0}: [epoch:{1:>2d}] [{2:>2d}/{3}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'BatchTime: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '
                    'lr: {lr:>4.6f} '.format(log_name,
                                             epoch + 1,
                                             step,
                                             steps_per_epoch,
                                             data_time=data_time_m,
                                             batch_time=batch_time_m,
                                             loss=losses_m,
                                             lr=learning_rate))

    return losses_m.avg


def validate(model, val_loader, device, logger, log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.eval()
    steps_per_epoch = len(val_loader)
    end = time.time()
    for step, batch in enumerate(val_loader):
        batch = tuple(t.to(device) for t in batch)
        outputs = model(**batch)
        loss = outputs[0]
        bs = loss.shape[0]

        data_time = time.time() - end
        it_time = time.time() - end
        end = time.time()

        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), bs)
        if (step % log_interval == 0) or (step == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                logger_name = 'Test-log'
                logger.info(
                    '{0}: [{1:>2d}/{2}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                        logger_name,
                        step,
                        steps_per_epoch,
                        data_time=data_time_m,
                        batch_time=batch_time_m,
                        loss=losses_m))
    return losses_m.avg


def predict(model, val_loader, device, logger, log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.eval()
    steps_per_epoch = len(val_loader)
    end = time.time()
    # Variables to gather full output
    true_labels, pred_labels = [], []
    for step, batch in enumerate(val_loader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, labels, token_types = batch
        outputs = model(**batch)
        loss = outputs[0]
        preds = outputs[1]
        preds = preds.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()

        true_labels.append(labels)
        pred_labels.append(preds)

        bs = loss.shape[0]
        data_time = time.time() - end
        it_time = time.time() - end
        end = time.time()

        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), bs)
        if (step % log_interval == 0) or (step == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                logger_name = 'Test-log'
                logger.info(
                    '{0}: [{1:>2d}/{2}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                        logger_name,
                        step,
                        steps_per_epoch,
                        data_time=data_time_m,
                        batch_time=batch_time_m,
                        loss=losses_m))
    # Flatten outputs
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    return true_labels, pred_labels


def train_loop(
    model,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_loader,
    val_loader,
    device,
    logger,
    start_epoch=0,
    end_epoch=0,
    early_stopping_patience=-1,
    save_checkpoints=True,
    checkpoint_dir='./',
    checkpoint_filename='checkpoint.pth.tar',
):
    is_best = True
    if early_stopping_patience > 0:
        epochs_since_improvement = 0

    best_prec1 = 0
    print(f'RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}')
    for epoch in range(start_epoch, end_epoch):
        losses_m = train(model,
                         train_loader,
                         optimizer,
                         lr_scheduler,
                         gradient_accumulation_steps,
                         epoch,
                         device,
                         logger,
                         log_interval=10)

        logger.info('[Epoch %d] training: loss=%f' % (epoch + 1, losses_m))
        losses_m = validate(model, val_loader, device, logger, log_interval=10)
        logger.info('[Epoch %d] validation: loss=%f' % (epoch + 1, losses_m))
        if save_checkpoints and (not torch.distributed.is_initialized()
                                 or torch.distributed.get_rank() == 0):
            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(
                checkpoint_state,
                is_best,
                checkpoint_dir=checkpoint_dir,
                filename=checkpoint_filename,
            )
        if early_stopping_patience > 0:
            if not is_best:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0
            if epochs_since_improvement >= early_stopping_patience:
                break
