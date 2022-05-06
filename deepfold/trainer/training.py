import time

import torch
from torch.autograd import Variable
from torch.cuda.amp import autocast

from deepfold.utils.metrics import AverageMeter
from deepfold.utils.model import reduce_tensor, save_checkpoint


def train(model,
          train_loader,
          optimizer,
          lr_scheduler,
          gradient_accumulation_steps,
          logger,
          epoch,
          log_interval=1):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.train()
    optimizer.zero_grad()
    steps_per_epoch = len(train_loader)
    end = time.time()
    batch_size = 1
    for step, batch in enumerate(train_loader):
        outputs = model(**batch)
        data_time = time.time() - end

        loss = outputs.loss

        loss = loss / gradient_accumulation_steps
        bs = loss.shape[0]
        if step % gradient_accumulation_steps == 0 or step == len(
                train_loader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        it_time = time.time() - end
        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), bs)

        end = time.time()
        if step == 0:
            batch_size = bs
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

    return losses_m.avg, batch_size


def validate(model,
             val_loader,
             criterion,
             logger,
             use_amp=False,
             log_interval=10):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')

    model.eval()
    steps_per_epoch = len(val_loader)
    end = time.time()
    batch_size = 1
    for step, batch in enumerate(val_loader):
        outputs = model(**batch)
        loss = outputs.loss
        bs = loss.shape[0]
        bs = input.size(0)
        data_time = time.time() - end
        it_time = time.time() - end
        end = time.time()

        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), bs)
        if step == 0:
            batch_size = bs
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
    save_checkpoints=True,
    checkpoint_dir='./',
    checkpoint_filename='checkpoint.pth.tar',
):
    is_best = True
    if early_stopping_patience > 0:
        epochs_since_improvement = 0

    print(f'RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}')
    for epoch in range(start_epoch, end_epoch):
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

        tic = time.time()
        losses_m, batch_size = validate(
            model,
            val_loader,
            criterion,
            logger,
            use_amp=use_amp,
        )
        steps_per_epoch = len(val_loader)
        throughput = int(batch_size * steps_per_epoch / (time.time() - tic))
        logger.info('[Epoch %d] validation: loss=%f' % (epoch + 1, losses_m))
        logger.info('[Epoch %d] speed: %d samples sec time cost: %f',
                    epoch + 1, throughput,
                    time.time() - tic)
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
