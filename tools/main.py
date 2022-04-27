import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader

sys.path.append('../')
from deepfold.data.protein_dataset import ProteinSequences
from deepfold.data.protein_tokenizer import ProteinTokenizer
from deepfold.models.lstm import LstmEncoderModel
from deepfold.scheduler import (CosineLRScheduler, ExponentialLRScheduler,
                                LinearLRScheduler, StepLRScheduler)
from deepfold.trainer.training import train_loop


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model-based Asynchronous HPO')
    parser.add_argument('--data_name',
                        default='',
                        type=str,
                        help='dataset name')
    parser.add_argument('--data_path',
                        default='',
                        type=str,
                        help='path to dataset')
    parser.add_argument('--data-backend', metavar='BACKEND', default='pytorch')
    parser.add_argument('--model',
                        metavar='MODEL',
                        default='resnet18',
                        help='model architecture: (default: resnet18)')
    parser.add_argument('-j',
                        '--workers',
                        type=int,
                        default=4,
                        metavar='N',
                        help='how many training processes to use (default: 1)')
    parser.add_argument('--epochs',
                        default=90,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument(
        '--early-stopping-patience',
        default=-1,
        type=int,
        metavar='N',
        help='early stopping after N epochs without improving',
    )
    parser.add_argument('-b',
                        '--batch-size',
                        default=256,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 256) per gpu')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--end-lr',
                        '--minimum learning-rate',
                        default=1e-8,
                        type=float,
                        metavar='END-LR',
                        help='initial learning rate')
    parser.add_argument(
        '--lr-schedule',
        default='step',
        type=str,
        metavar='SCHEDULE',
        choices=['step', 'linear', 'cosine', 'exponential'],
        help='Type of LR schedule: {}, {}, {} , {}'.format(
            'step', 'linear', 'cosine', 'exponential'),
    )
    parser.add_argument('--warmup',
                        default=0,
                        type=int,
                        metavar='E',
                        help='number of warmup epochs')
    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        choices=('sgd', 'rmsprop', 'adamw'))
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--log_interval',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--training-only',
                        action='store_true',
                        help='do not evaluate')
    parser.add_argument(
        '--no-checkpoints',
        action='store_false',
        dest='save_checkpoints',
        help='do not store any checkpoints, useful for benchmarking',
    )
    parser.add_argument('--checkpoint-filename',
                        default='checkpoint.pth.tar',
                        type=str)
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp',
                        action='store_true',
                        default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp',
                        action='store_true',
                        default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument(
        '--static-loss-scale',
        type=float,
        default=1,
        help='Static loss scale',
    )
    parser.add_argument(
        '--dynamic-loss-scale',
        action='store_true',
        help='Use dynamic loss scaling.  If supplied, this argument supersedes '
        + '--static-loss-scale.',
    )
    parser.add_argument('--output-dir',
                        default='./work_dirs/protein_function_go',
                        type=str,
                        help='output directory for model and log')
    args = parser.parse_args()
    return args


def prepare_for_training(args):
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if args.seed is not None:
        print('Using seed = {}'.format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)

    else:

        def _worker_init_fn(id):
            pass

    if args.static_loss_scale != 1.0:
        if not args.amp:
            print(
                'Warning: if --amp is not used, static_loss_scale will be ignored.'
            )

    start_epoch = 0
    # Create data loaders
    # Result file with a list of terms for prediction task
    terms_file = os.path.join(args.data_path, u'terms.pkl')
    # Result file with a DataFrame for training
    train_data_file = os.path.join(args.data_path, u'train_data.pkl')
    # Result file with a DataFrame for testing
    test_data_file = os.path.join(args.data_path, u'test_data.pkl')
    # Result file with a DataFrame of prediction annotations
    predictions_file = os.path.join(args.data_path, 'predictions.pkl')

    data_path_dict = {}
    data_path_dict['terms'] = terms_file
    data_path_dict['train_data'] = train_data_file
    data_path_dict['test_data'] = test_data_file
    data_path_dict['predictions'] = predictions_file

    # Terms of annotations
    terms_df = pd.read_pickle(data_path_dict['terms'])
    terms = terms_df['terms'].values.flatten()
    num_classes = len(terms)

    # get data loaders
    # Dataset and DataLoader
    train_dataset = ProteinSequences(train_data_file, terms_file)
    test_dataset = ProteinSequences(test_data_file, terms_file)

    # dataloders
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)
    valid_loader = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)
    # test_loader = DataLoader(test_dataset,
    #                          batch_size=args.batch_size,
    #                          collate_fn=train_dataset.collate_fn,
    #                          shuffle=False)

    # model
    vocab_size = ProteinTokenizer().vocab_size
    model = LstmEncoderModel(vocab_size=vocab_size,
                             embed_dim=128,
                             hidden_size=1024,
                             num_labels=num_classes,
                             bidirectional=False)

    # model
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], output_device=args.gpu)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              output_device=0)
    else:
        model.cuda()

    # define loss function (criterion) and optimizer
    # Creat train losses
    criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    # optimizer and lr_policy
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.lr_schedule == 'step':
        lr_policy = StepLRScheduler(optimizer=optimizer,
                                    base_lr=args.lr,
                                    steps=[30, 60, 80],
                                    decay_factor=0.1,
                                    warmup_length=args.warmup,
                                    logger=logger)
    elif args.lr_schedule == 'cosine':
        lr_policy = CosineLRScheduler(optimizer=optimizer,
                                      base_lr=args.lr,
                                      warmup_length=args.warmup,
                                      epochs=args.epochs,
                                      end_lr=args.end_lr,
                                      logger=logger)
    elif args.lr_schedule == 'linear':
        lr_policy = LinearLRScheduler(optimizer=optimizer,
                                      base_lr=args.lr,
                                      warmup_length=args.warmup,
                                      epochs=args.epochs,
                                      logger=logger)
    elif args.lr_schedule == 'exponential':
        lr_policy = ExponentialLRScheduler(optimizer=optimizer,
                                           base_lr=args.lr,
                                           warmup_length=args.warmup,
                                           epochs=args.epochs,
                                           logger=logger)

    scaler = torch.cuda.amp.GradScaler(
        init_scale=args.static_loss_scale,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=100 if args.dynamic_loss_scale else 1000000000,
        enabled=args.amp,
    )

    return (model, criterion, optimizer, lr_policy, scaler, train_loader,
            valid_loader, start_epoch)


def main(args):
    global best_prec1
    best_prec1 = 0
    model, criterion, optimizer, lr_policy, scaler, train_loader, valid_loader, start_epoch = prepare_for_training(
        args)
    train_loop(
        model,
        criterion,
        optimizer,
        scaler,
        lr_policy,
        train_loader,
        valid_loader,
        logger=logger,
        use_amp=args.amp,
        start_epoch=start_epoch,
        end_epoch=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        best_prec1=best_prec1,
        skip_training=args.evaluate,
        skip_validation=args.training_only,
        save_checkpoints=args.save_checkpoints and not args.evaluate,
        checkpoint_dir=args.output_dir,
        checkpoint_filename=args.checkpoint_filename,
    )
    print('Experiment ended')


if __name__ == '__main__':
    args = parse_args()
    task_name = args.data_name + '-' + args.model
    args.output_dir = os.path.join(args.output_dir, task_name)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank(
    ) == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    logger = logging.getLogger('')
    filehandler = logging.FileHandler(
        os.path.join(args.output_dir, 'summary.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    cudnn.benchmark = True
    start_time = time.time()
    main(args)
