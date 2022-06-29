import argparse
import logging
import os
import sys

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from deepfold.data.seq2vec_dataset import Seq2VecDataset
from deepfold.models.seq2vec_model import Seq2VecEmbedder
from deepfold.trainer.embeds import extract_esm_embedds

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
                    default='seq2vec',
                    help='model architecture: (default: seq2vec)')
parser.add_argument('--pool_mode',
                    metavar='MODEL',
                    default='mean',
                    help='embedding method')
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


def main(args):
    if args.split == 'train':
        data_file = os.path.join(args.data_path, 'train_data.pkl')
        file_name = 'train_data.pkl'
    else:
        data_file = os.path.join(args.data_path, 'test_data.pkl')
        file_name = 'test_data.pkl'

    model_dir = os.path.join(args.data_path, 'seq2vec')
    assert os.path.exists(data_file)
    save_path = os.path.join(
        args.data_path, args.model + '_embeddings_' + args.pool_mode + '_' +
        args.split + '.pkl')
    print(
        'Pretrained model %s, pool_mode: %s,  data split: %s , file path: %s' %
        (args.model, args.pool_mode, args.split, data_file))
    print('Embeddings save path: ', save_path)
    # Dataset and DataLoader
    dataset = Seq2VecDataset(data_path=args.data_path, file_name=file_name)
    # dataloders
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             collate_fn=dataset.collate_fn,
                             pin_memory=True)
    # model
    num_labels = dataset.num_classes
    model = Seq2VecEmbedder(model_dir=model_dir,
                            pool_mode=args.pool_mode,
                            num_labels=num_labels)
    # run predict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    embeddings, true_labels = extract_esm_embedds(model,
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
