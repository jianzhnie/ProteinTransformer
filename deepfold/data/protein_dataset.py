import os
import random
import re
import sys

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, RobertaTokenizer

from deepfold.data.protein_tokenizer import ProteinTokenizer

sys.path.append('../../')


class ProtRobertaDataset(Dataset):
    def __init__(self,
                 data_path='dataset/',
                 tokenizer_dir='tokenizer/',
                 split='train',
                 max_length=1024):
        self.datasetFolderPath = data_path
        self.trainFilePath = os.path.join(self.datasetFolderPath,
                                          'train_data.pkl')
        self.testFilePath = os.path.join(self.datasetFolderPath,
                                         'test_data.pkl')
        self.termsFilePath = os.path.join(self.datasetFolderPath, 'terms.pkl')

        # load pre-trained tokenizer
        self.tokenizer = RobertaTokenizer(
            vocab_file=os.path.join(tokenizer_dir, 'vocab.json'),
            merges_file=os.path.join(tokenizer_dir, 'merges.txt'),
            local_files_only=True)

        if split == 'train':
            self.seqs, self.labels, self.terms = self.load_dataset(
                self.trainFilePath, self.termsFilePath)
        else:
            self.seqs, self.labels, self.terms = self.load_dataset(
                self.testFilePath, self.termsFilePath)

        self.num_classes = len(self.terms)
        self.max_length = max_length
        self.id2label = {idx: label for idx, label in enumerate(self.terms)}
        self.label2id = {label: idx for idx, label in enumerate(self.terms)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # Make sure there is a space between every token, and map rarely amino acids
        sequence = self.seqs[idx]
        length = len(sequence)

        seq_ids = self.tokenizer(sequence,
                                 padding='max_length',
                                 max_length=self.max_length,
                                 truncation=True)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}

        label_list = self.labels[idx]
        multilabel = [0] * self.num_classes
        for t_id in label_list:
            if t_id in self.label2id:
                label_idx = self.label2id[t_id]
                multilabel[label_idx] = 1

        sample['labels'] = torch.tensor(multilabel, dtype=torch.int)
        sample['lengths'] = torch.tensor(length, dtype=torch.int)
        return sample

    def load_dataset(self, data_path, term_path):
        df = pd.read_pickle(data_path)
        terms_df = pd.read_pickle(term_path)
        terms = terms_df['terms'].values.flatten()

        seq = list(df['sequences'])
        label = list(df['prop_annotations'])
        assert len(seq) == len(label)
        return seq, label, terms


class ProtBertDataset(Dataset):
    def __init__(self,
                 data_path='dataset/',
                 split='train',
                 tokenizer_name='Rostlab/prot_bert_bfd',
                 max_length=1024):
        self.datasetFolderPath = data_path
        self.trainFilePath = os.path.join(self.datasetFolderPath,
                                          'train_data.pkl')
        self.testFilePath = os.path.join(self.datasetFolderPath,
                                         'test_data.pkl')
        self.termsFilePath = os.path.join(self.datasetFolderPath, 'terms.pkl')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                       do_lower_case=False)

        if split == 'train':
            self.seqs, self.labels, self.terms = self.load_dataset(
                self.trainFilePath, self.termsFilePath)
        else:
            self.seqs, self.labels, self.terms = self.load_dataset(
                self.testFilePath, self.termsFilePath)

        self.num_classes = len(self.terms)
        self.max_length = max_length
        self.id2label = {idx: label for idx, label in enumerate(self.terms)}
        self.label2id = {label: idx for idx, label in enumerate(self.terms)}

    def load_dataset(self, data_path, term_path):
        df = pd.read_pickle(data_path)
        terms_df = pd.read_pickle(term_path)
        terms = terms_df['terms'].values.flatten()

        seq = list(df['sequences'])
        label = list(df['prop_annotations'])
        assert len(seq) == len(label)
        return seq, label, terms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # Make sure there is a space between every token, and map rarely amino acids
        seq = ' '.join(''.join(self.seqs[idx].split()))
        seq = re.sub(r'[UZOB]', 'X', seq)

        seq_ids = self.tokenizer(
            seq,
            # add_special_tokens=True,  #Add [CLS] [SEP] tokens
            padding='max_length',
            max_length=self.max_length,
            truncation=True,  # Truncate data beyond max length
            # return_token_type_ids=False,
            # return_attention_mask=True,  # diff normal/pad tokens
            # return_tensors='pt'  # PyTorch Tensor format
        )

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}

        label_list = self.labels[idx]
        multilabel = [0] * self.num_classes
        for t_id in label_list:
            if t_id in self.label2id:
                label_idx = self.label2id[t_id]
                multilabel[label_idx] = 1

        sample['labels'] = torch.tensor(multilabel)
        return sample


class ProtSeqDataset(Dataset):
    def __init__(self,
                 data_path: str = 'dataset/',
                 split: str = 'train',
                 max_length: int = 1024,
                 truncate: bool = True,
                 random_crop: bool = False):
        super().__init__()

        self.datasetFolderPath = data_path
        self.trainFilePath = os.path.join(self.datasetFolderPath,
                                          'train_data.pkl')
        self.testFilePath = os.path.join(self.datasetFolderPath,
                                         'test_data.pkl')
        self.termsFilePath = os.path.join(self.datasetFolderPath, 'terms.pkl')

        if split == 'train':
            self.seqs, self.labels, self.terms = self.load_dataset(
                self.trainFilePath, self.termsFilePath)
        else:
            self.seqs, self.labels, self.terms = self.load_dataset(
                self.testFilePath, self.termsFilePath)

        # self.
        self.tokenizer = ProteinTokenizer()
        self.num_classes = len(self.terms)
        self.max_length = max_length
        self.truncate = truncate
        self.random_crop = random_crop
        self.id2label = {idx: label for idx, label in enumerate(self.terms)}
        self.label2id = {label: idx for idx, label in enumerate(self.terms)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sequence = self.seqs[idx]
        if self.random_crop:
            sequence = crop_sequence(sequence, crop_length=self.max_length - 2)
        if self.truncate:
            sequence = sequence[:self.max_length - 2]

        length = len(sequence)
        multilabel = [0] * self.num_classes
        anno_term_list = self.labels[idx]
        for t_id in anno_term_list:
            if t_id in self.label2id:
                label_idx = self.label2id[t_id]
                multilabel[label_idx] = 1

        token_ids = self.tokenizer.gen_token_ids(sequence)
        return token_ids, length, multilabel

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs = [torch.tensor(ex[0]) for ex in examples]
        lengths = [ex[1] for ex in examples]
        targets = [ex[2] for ex in examples]

        # 对batch内的样本进行padding，使其具有相同长度
        inputs = pad_sequence(inputs,
                              batch_first=True,
                              padding_value=self.tokenizer.padding_token_id)
        encoded_inputs = {'input_ids': inputs}
        encoded_inputs['lengths'] = torch.tensor(lengths, dtype=torch.int)
        encoded_inputs['labels'] = torch.tensor(targets, dtype=torch.int)
        return encoded_inputs

    def load_dataset(self, data_path, term_path):
        df = pd.read_pickle(data_path)
        terms_df = pd.read_pickle(term_path)
        terms = terms_df['terms'].values.flatten()

        prot_seqs = list(df['sequences'])
        anno_terms = list(df['prop_annotations'])
        assert len(prot_seqs) == len(anno_terms)
        return prot_seqs, anno_terms, terms


def crop_sequence(sequence: str, crop_length: int) -> str:
    """If the length of the sequence is superior to crop_length, crop randomly
    the sequence to get the proper length."""
    if len(sequence) <= crop_length:
        return sequence
    else:
        start_idx = random.randint(0, len(sequence) - crop_length)
        return sequence[start_idx:(start_idx + crop_length)]


if __name__ == '__main__':

    pro_dataset = ProtSeqDataset(
        data_path='/Users/robin/xbiome/datasets/protein', split=True)
    for i in range(10):
        sample = pro_dataset[i]
        print(sample)

    pro_dataset = ProtBertDataset(
        data_path='/Users/robin/xbiome/datasets/protein')
    print(pro_dataset.num_classes)
    for i in range(10):
        sample = pro_dataset[i]
        for key, val in sample.items():
            print(key, val)
