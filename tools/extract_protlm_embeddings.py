import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transformers import RobertaConfig

from deepfold.data.protein_dataset import ProtRobertaDataset
from deepfold.models.transformers.multilabel_transformer import \
    RobertaForMultiLabelSequenceClassification
from deepfold.trainer.embeds import extract_seq_embedds

sys.path.append('../')

parser = argparse.ArgumentParser(
    description='Protein function Classification Model Train config')
parser.add_argument('--data_path',
                    default='',
                    type=str,
                    help='data dir of dataset')
parser.add_argument('--pretrain_model_dir',
                    default='',
                    type=str,
                    help='pretrained model checkpoint dir')
parser.add_argument('--split',
                    default='train',
                    help=' train or test data split')
parser.add_argument('--model',
                    metavar='MODEL',
                    default='bert',
                    help='model architecture: (default: bert)')
parser.add_argument('--pool_mode', default='mean', help='embedding method')
parser.add_argument('--fintune', default=True, type=bool, help='fintune model')
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


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    # vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu


def main(args):
    model_name = 'roberta'
    if args.split == 'train':
        data_file = os.path.join(args.data_path, 'train_data.pkl')
    else:
        data_file = os.path.join(args.data_path, 'test_data.pkl')

    assert os.path.exists(data_file)
    save_path = os.path.join(
        args.data_path, model_name + '_embeddings_' + args.pool_mode + '_' +
        args.split + '.pkl')
    print(
        'Pretrained model %s, pool_mode: %s,  data split: %s , file path: %s' %
        (model_name, args.pool_mode, args.split, data_file))
    print('Embeddings save path: ', save_path)
    # Dataset and DataLoader
    dataset = ProtRobertaDataset(data_path=args.data_path,
                                 tokenizer_dir=args.pretrain_model_dir,
                                 split=args.split,
                                 max_length=1024)
    # dataloders
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True)
    # model
    num_classes = dataset.num_classes
    model_config = RobertaConfig.from_pretrained(
        pretrained_model_name_or_path=args.pretrain_model_dir,
        num_labels=num_classes)
    model = RobertaForMultiLabelSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.pretrain_model_dir,
        config=model_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # run predict
    embeddings, true_labels = extract_seq_embedds(model,
                                                  data_loader,
                                                  pool_mode=args.pool_mode,
                                                  logger=logger,
                                                  device=device)
    print(embeddings.shape, true_labels.shape)
    df = pd.read_pickle(data_file)
    df['esm_embeddings'] = embeddings.tolist()
    df['labels'] = true_labels.tolist()
    df.to_pickle(save_path)
    print('Embeddings saved to :', save_path)


if __name__ == '__main__':
    logger = logging.getLogger('')
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(streamhandler)
    args = parser.parse_args()
    cudnn.benchmark = True
    main(args)
