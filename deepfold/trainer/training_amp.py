import time

import numpy as np
import torch

from deepfold.utils.metrics import AverageMeter
from deepfold.utils.model import reduce_tensor, save_checkpoint

try:
    from apex import amp
except ImportError:
    raise ImportError(
        'Please install apex from https://www.github.com/nvidia/apex to run this example.'
    )


def train(model,
          loader,
          optimizer,
          lr_scheduler,
          epoch,
          logger,
          log_interval=1):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.train()
    steps_per_epoch = len(loader)
    end = time.time()
    for idx, batch in enumerate(loader):
        lr_scheduler.step(epoch)
        # Add batch to GPU
        batch = {key: val.cuda() for key, val in batch.items()}

        data_time = time.time() - end

        outputs = model(**batch)
        loss = outputs[0]

        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        if torch.distributed.is_initialized():
            reduced_loss = reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        batch_size = batch['input_ids'].shape[0]

        it_time = time.time() - end
        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(reduced_loss.item(), batch_size)

        end = time.time()
        if (idx % log_interval == 0) or (idx == steps_per_epoch - 1):
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
                                             idx,
                                             steps_per_epoch,
                                             data_time=data_time_m,
                                             batch_time=batch_time_m,
                                             loss=losses_m,
                                             lr=learning_rate))

    return losses_m.avg


def validate(model, val_loader, logger, log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.eval()
    steps_per_epoch = len(val_loader)
    end = time.time()
    for idx, batch in enumerate(val_loader):
        batch = {key: val.cuda() for key, val in batch.items()}

        data_time = time.time() - end

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs[0]

        if torch.distributed.is_initialized():
            reduced_loss = reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        batch_size = batch['input_ids'].shape[0]

        it_time = time.time() - end
        end = time.time()

        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(reduced_loss.item(), batch_size)
        if (idx % log_interval == 0) or (idx == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                logger_name = 'Test-log'
                logger.info(
                    '{0}: [{1:>2d}/{2}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                        logger_name,
                        idx,
                        steps_per_epoch,
                        data_time=data_time_m,
                        batch_time=batch_time_m,
                        loss=losses_m))
    return losses_m.avg


def predict(model, val_loader, logger, log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.eval()
    steps_per_epoch = len(val_loader)
    end = time.time()
    # Variables to gather full output
    true_labels, pred_labels = [], []
    for idx, batch in enumerate(val_loader):
        batch = {key: val.cuda() for key, val in batch.items()}
        labels = batch['labels']
        outputs = model(**batch)
        loss = outputs[0]
        logits = outputs[1]
        preds = torch.sigmoid(logits)
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
        if (idx % log_interval == 0) or (idx == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                logger_name = 'Test-log'
                logger.info(
                    '{0}: [{1:>2d}/{2}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                        logger_name,
                        idx,
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
    train_loader,
    val_loader,
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

    best_loss = np.inf
    print(f'RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}')
    for epoch in range(start_epoch, end_epoch):
        train_loss = train(model,
                           train_loader,
                           optimizer,
                           lr_scheduler,
                           epoch,
                           logger,
                           log_interval=10)

        logger.info('[Epoch %d] training: loss=%f' % (epoch + 1, train_loss))
        val_loss = validate(model, val_loader, logger, log_interval=10)
        logger.info('[Epoch %d] validation: loss=%f' % (epoch + 1, val_loss))

        if train_loss < best_loss:
            is_best = True
            best_loss = train_loss

        if save_checkpoints and (not torch.distributed.is_initialized()
                                 or torch.distributed.get_rank() == 0):
            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': val_loss,
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
