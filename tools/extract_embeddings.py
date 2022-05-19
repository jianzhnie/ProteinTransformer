import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import DataLoader

from deepfold.data.esm_dataset import ESMDataset
from deepfold.models.esm_model import ESMTransformer

sys.path.append('../')

parser = argparse.ArgumentParser(
    description='Protein function Classification Model Train config')
parser.add_argument('--data_path',
                    default='',
                    type=str,
                    help='data dir of dataset')
parser.add_argument('--split',
                    default='train',
                    help=' train or test data split')
parser.add_argument('--model',
                    metavar='MODEL',
                    default='esm',
                    help='model architecture: (default: esm)')
parser.add_argument('--pool_mode',
                    metavar='MODEL',
                    default='mean',
                    help='embedding method')
parser.add_argument('--fintune', default=True, type=bool, help='fintune model')
parser.add_argument('--local_rank', default=0, type=int)
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
parser.add_argument('--output-dir',
                    default='./work_dirs',
                    type=str,
                    help='output directory for model and log')


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


def GetModelEmbedding(model, data_loader, pool_mode):
    embeddings = []
    true_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):

            if torch.cuda.is_available():
                batch = {
                    key: val.to(device='cuda')
                    for key, val in batch.items()
                }
            labels = batch['labels']
            embeddings_dict = model.compute_embeddings(**batch)
            batch_embeddings = embeddings_dict[pool_mode].to(
                device='cpu').numpy()
            labels = labels.to('cpu').numpy()
            true_labels.append(labels)
            embeddings.append(batch_embeddings)
            print(
                f"Processing {batch_idx + 1} of {len(data_loader)} batches ({batch['input_ids'].size(0)} sequences)"
            )
    embeddings = np.concatenate(embeddings, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    return embeddings, true_labels


def main(args):
    model_name = 'esm1b_t33_650M_UR50S'
    if args.split == 'train':
        data_file = os.path.join(args.data_path, 'train_data.pkl')
    else:
        data_file = os.path.join(args.data_path, 'test_data.pkl')

    # Dataset and DataLoader
    dataset = ESMDataset(data_path=args.data_path,
                         split=args.split,
                         model_dir=model_name)
    # dataloders
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             collate_fn=dataset.collate_fn,
                             pin_memory=True)
    # model
    num_labels = dataset.num_classes
    model = ESMTransformer(model_dir=model_name,
                           pool_mode=args.pool_mode,
                           fintune=args.fintune,
                           num_labels=num_labels)
    model = model.cuda()
    # run predict
    embeddings, true_labels = GetModelEmbedding(model,
                                                data_loader,
                                                pool_mode=args.pool_mode)
    print(embeddings.shape, true_labels.shape)

    df = pd.read_pickle(data_file)
    df['esm_embeddings'] = embeddings.tolist()
    df['labels'] = true_labels.tolist()
    save_path = os.path.join(
        args.data_path,
        model_name + '_last_mean_embedding_' + args.split + '.pkl')
    print(save_path)
    df.to_pickle(save_path)


if __name__ == '__main__':
    args = parser.parse_args()
    args.work_dir = os.path.join(args.output_dir, args.model)
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    cudnn.benchmark = True

    main(args)
