import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast
from tqdm import tqdm

from deepfold.utils.metrics import AverageMeter
from deepfold.utils.model import reduce_tensor, save_checkpoint


def do_compute(model, batch):
    logits = model(*batch[:-1])
    return logits, batch[-1]


def get_train_step(model, criterion, optimizer, scaler, use_amp=False):
    def _step(input, target, optimizer_step=True):
        input_var = Variable(input)
        target_var = Variable(target)

        with autocast(enabled=use_amp):
            output = model(input_var)
            loss = criterion(output, target_var)

            if torch.distributed.is_initialized():
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

        scaler.scale(loss).backward()

        if optimizer_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        return reduced_loss

    return _step


def train(model,
          train_loader,
          criterion,
          optimizer,
          scaler,
          lr_scheduler,
          logger,
          epoch,
          use_amp=False,
          log_interval=1):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    step = get_train_step(model,
                          criterion,
                          optimizer,
                          scaler=scaler,
                          use_amp=use_amp)

    model.train()
    optimizer.zero_grad()
    steps_per_epoch = len(train_loader)
    end = time.time()
    batch_size = 1
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        bs = input.size(0)
        lr_scheduler.step(epoch)
        data_time = time.time() - end

        loss = step(input, target)

        it_time = time.time() - end
        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), bs)

        end = time.time()
        if i == 1:
            batch_size = bs
        if (i % log_interval == 0) or (i == steps_per_epoch - 1):
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
                                             i,
                                             steps_per_epoch,
                                             data_time=data_time_m,
                                             batch_time=batch_time_m,
                                             loss=losses_m,
                                             lr=learning_rate))

    return losses_m.avg, batch_size


def get_val_step(model, criterion, use_amp=False):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad(), autocast(enabled=use_amp):
            output = model(input_var)
            loss = criterion(output, target_var)

            if torch.distributed.is_initialized():
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss

    return _step


def validate(model,
             val_loader,
             criterion,
             logger,
             use_amp=False,
             log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    step = get_val_step(model, criterion, use_amp=use_amp)
    # switch to evaluate mode
    model.eval()
    steps_per_epoch = len(val_loader)
    end = time.time()
    batch_size = 1
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        bs = input.size(0)
        data_time = time.time() - end
        loss = step(input, target)
        it_time = time.time() - end
        end = time.time()

        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), bs)

        if i == 1:
            batch_size = bs

        if (i % log_interval == 0) or (i == steps_per_epoch - 1):
            if not torch.distributed.is_initialized(
            ) or torch.distributed.get_rank() == 0:
                logger_name = 'Test-log'
                logger.info(
                    '{0}: [{1:>2d}/{2}] '
                    'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                        logger_name,
                        i,
                        steps_per_epoch,
                        data_time=data_time_m,
                        batch_time=batch_time_m,
                        loss=losses_m))
    return losses_m.avg, batch_size


def train_loop(
    model,
    criterion,
    optimizer,
    scaler,
    lr_scheduler,
    train_loader,
    val_loader,
    logger,
    use_amp=False,
    best_prec1=0,
    start_epoch=0,
    end_epoch=0,
    early_stopping_patience=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir='./',
    checkpoint_filename='checkpoint.pth.tar',
):
    prec1 = -1

    if early_stopping_patience > 0:
        epochs_since_improvement = 0

    print(f'RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}')
    for epoch in range(start_epoch, end_epoch):
        if not skip_training:
            tic = time.time()
            losses_m, batch_size = train(model,
                                         train_loader,
                                         criterion,
                                         optimizer,
                                         scaler,
                                         lr_scheduler,
                                         logger,
                                         epoch,
                                         use_amp=use_amp,
                                         log_interval=10)

        steps_per_epoch = len(train_loader)
        throughput = int(batch_size * steps_per_epoch / (time.time() - tic))
        logger.info('[Epoch %d] training: loss=%f' % (epoch + 1, losses_m))
        logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f',
                    epoch + 1, throughput,
                    time.time() - tic)

        if not skip_validation:
            tic = time.time()
            losses_m, batch_size = validate(
                model,
                val_loader,
                criterion,
                logger,
                use_amp=use_amp,
            )
            steps_per_epoch = len(val_loader)
            throughput = int(batch_size * steps_per_epoch /
                             (time.time() - tic))
            logger.info('[Epoch %d] validation: loss=%f' %
                        (epoch + 1, losses_m))
            logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f',
                        epoch + 1, throughput,
                        time.time() - tic)
            if prec1 > best_prec1:
                is_best = True
                best_prec1 = prec1
            else:
                is_best = False
        else:
            is_best = True
            best_prec1 = 0

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


def run_batch(model, optimizer, data_loader, epoch_i, desc, loss_fn):
    total_loss = 0
    logits_list = []
    ground_truth = []

    for batch in tqdm(data_loader, desc=f'{desc} Epoch {epoch_i}'):
        logits, labels = do_compute(model, batch)

        loss = loss_fn(logits, labels)
        loss = np.mean(np.sum(loss, -1))
        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        total_loss += loss.item()

        logits_list.append(F.sigmoid(logits).tolist())
        ground_truth.append(labels.tolist())

    total_loss /= len(data_loader)

    logits_list = np.concatenate(logits_list)
    ground_truth = np.concatenate(ground_truth)

    metrics = None
    return total_loss, metrics
