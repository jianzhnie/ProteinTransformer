import argparse
import logging
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transformers import AutoModel
sys.path.append('../')
from deepfold.data.ontotextual_dataset import OntoTextDataset
from deepfold.trainer.embeds import extract_sentence_embedds


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


def main(args):
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    data_file = os.path.join(args.data_path, 'GotermText.csv')
    assert os.path.exists(data_file)
    save_path = os.path.join(
        args.data_path, 'onto_embeddings_' + args.pool_mode + '_' + args.split + '.pkl')
    print('Pretrained model %s, pool_mode: %s,  file path: %s' %
          (model_name, args.pool_mode, data_file))
    print('Embeddings save path: ', save_path)
    # Dataset and DataLoader
    dataset = OntoTextDataset(data_dir=args.data_path,
                              tokenizer_name=model_name,
                              max_length=512)

    # dataloders
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True)
    # model
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # run predict
    embeddings = extract_sentence_embedds(model,
                                          data_loader,
                                          pool_mode=args.pool_mode,
                                          logger=logger,
                                          device=device)
    print(embeddings.shape)


    df = pd.read_csv(data_file)
    df['embeddings'] = embeddings.tolist()
    df.to_csv(save_path)

    print('Embeddings saved to :', save_path)


if __name__ == '__main__':
    logger = logging.getLogger('')
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(streamhandler)
    args = parser.parse_args()
    cudnn.benchmark = True
    main(args)
