import argparse
import logging
import os
import sys
import time

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import yaml
from torch.utils.data import DataLoader

from deepfold.data.esm_dataset import ESMDataset
from deepfold.models.esm_model import ESMTransformer
from deepfold.trainer.training import Predict
from deepfold.utils.model import load_model_checkpoint

sys.path.append('../')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config',
                                                 add_help=False)
parser.add_argument('-c',
                    '--config',
                    default='',
                    type=str,
                    metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(
    description='Protein function Classification Model Train config')
parser.add_argument('--data_path',
                    default='',
                    type=str,
                    help='data dir of dataset')
parser.add_argument('--model',
                    metavar='MODEL',
                    default='esm',
                    help='model architecture: (default: esm)')
parser.add_argument('--pool_mode',
                    metavar='MODEL',
                    default='mean',
                    help='embedding method')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-j',
                    '--workers',
                    type=int,
                    default=4,
                    metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256) per gpu')


def main(args):
    # Dataset and DataLoader
    test_dataset = ESMDataset(data_path=args.data_path,
                              split='test',
                              model_dir='esm1b_t33_650M_UR50S')
    # dataloders
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             collate_fn=test_dataset.collate_fn,
                             pin_memory=True)

    # model
    num_labels = test_dataset.num_classes
    model = ESMTransformer(model_dir='esm1b_t33_650M_UR50S',
                           pool_mode=args.pool_mode,
                           num_labels=num_labels)

    if args.resume is not None:
        if args.local_rank == 0:
            model_state, optimizer_state = load_model_checkpoint(
                args.resume, args)
            model.load_state_dict(model_state)

    # define loss function (criterion) and optimizer
    # optimizer and lr_policy
    criterion = nn.BCEWithLogitsLoss().cuda()
    model = model.cuda()

    # run predict
    predictions, test_metrics = Predict(model,
                                        test_loader,
                                        criterion,
                                        use_amp=args.amp,
                                        logger=logger)

    logger.info('Test metrics: %s' % (test_metrics))

    test_data_path = os.path.join(args.data_path, 'test_data.pkl')
    test_df = pd.read_pickle(test_data_path)

    preds, test_labels = predictions
    test_df['labels'] = list(test_labels)
    test_df['preds'] = list(preds)
    df_path = os.path.join(args.data_path, 'predictions.pkl')
    test_df.to_pickle(df_path)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == '__main__':
    args, args_text = _parse_args()
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    task_name = 'ProtLM' + '_' + args.model
    args.output_dir = os.path.join(args.output_dir, task_name)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank(
    ) == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

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
