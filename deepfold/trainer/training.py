import time

import numpy as np
import torch
from torch.cuda.amp import autocast

from deepfold.utils.metrics import AverageMeter
from deepfold.utils.model import reduce_tensor, save_checkpoint


def get_train_step(model,
                   optimizer,
                   scaler,
                   gradient_accumulation_steps,
                   use_amp=False):
    def _step(**inputs):

        with autocast(enabled=use_amp):
            outputs = model(**inputs)
            loss = outputs[0]
            loss /= gradient_accumulation_steps
            if torch.distributed.is_initialized():
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        torch.cuda.synchronize()

        return reduced_loss

    return _step


def train(model,
          loader,
          optimizer,
          scaler,
          lr_scheduler,
          gradient_accumulation_steps,
          epoch,
          use_amp,
          device,
          logger,
          log_interval=1):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    step = get_train_step(model, optimizer, scaler,
                          gradient_accumulation_steps, use_amp)

    model.train()
    optimizer.zero_grad()
    steps_per_epoch = len(loader)
    end = time.time()
    for idx, batch in enumerate(loader):
        lr_scheduler.step(epoch)
        # Add batch to GPU
        batch = {key: val.to(device) for key, val in batch.items()}

        data_time = time.time() - end

        loss = step(**batch)
        batch_size = batch['input_ids'].shape[0]

        it_time = time.time() - end
        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), batch_size)

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


def get_val_step(model, use_amp=False):
    def _step(**inputs):

        with autocast(enabled=use_amp):
            outputs = model(**inputs)
            loss = outputs[0]
            if torch.distributed.is_initialized():
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss

    return _step


def validate(model, val_loader, use_amp, device, logger, log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    step = get_val_step(model, use_amp)

    model.eval()
    steps_per_epoch = len(val_loader)
    end = time.time()
    for idx, batch in enumerate(val_loader):
        batch = {key: val.to(device) for key, val in batch.items()}

        data_time = time.time() - end

        loss = step(**batch)
        batch_size = batch['input_ids'].shape[0]

        it_time = time.time() - end
        end = time.time()

        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), batch_size)
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


def predict(model, val_loader, device, logger, log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.eval()
    steps_per_epoch = len(val_loader)
    end = time.time()
    # Variables to gather full output
    true_labels, pred_labels = [], []
    for idx, batch in enumerate(val_loader):
        batch = {key: val.to(device) for key, val in batch.items()}
        labels = batch['labels']
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
    scaler,
    gradient_accumulation_steps,
    train_loader,
    val_loader,
    use_amp,
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
                         scaler,
                         lr_scheduler,
                         gradient_accumulation_steps,
                         epoch,
                         use_amp,
                         device,
                         logger,
                         log_interval=10)

        logger.info('[Epoch %d] training: loss=%f' % (epoch + 1, losses_m))
        losses_m = validate(model,
                            val_loader,
                            use_amp,
                            device,
                            logger,
                            log_interval=10)
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
