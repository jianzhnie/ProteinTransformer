import os
import re

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .protein_tokenizer import ProteinTokenizer


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

        self.terms_dict = {v: i for i, v in enumerate(self.terms)}
        self.num_classes = len(self.terms)
        self.max_length = max_length

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
            if t_id in self.terms_dict:
                label_idx = self.terms_dict[t_id]
                multilabel[label_idx] = 1

        sample['labels'] = torch.tensor(multilabel)
        return sample


class CustomProteinSequences(Dataset):
    def __init__(self, data_path, split='train', max_length=1024):
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
        self.terms_dict = {v: i for i, v in enumerate(self.terms)}
        self.num_classes = len(self.terms)
        self.max_length = max_length

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        seqence = self.seqs[idx]
        label_list = self.labels[idx]

        multilabel = [0] * self.num_classes
        for t_id in label_list:
            if t_id in self.terms_dict:
                label_idx = self.terms_dict[t_id]
                multilabel[label_idx] = 1

        token_ids = self.tokenizer.gen_token_ids(seqence)
        return token_ids, multilabel

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs = [torch.tensor(ex[0]) for ex in examples]
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.float)
        # 对batch内的样本进行padding，使其具有相同长度
        inputs = pad_sequence(inputs,
                              batch_first=True,
                              padding_value=self.tokenizer.padding_token_id)

        return (inputs, targets)

    def load_dataset(self, data_path, term_path):
        df = pd.read_pickle(data_path)
        terms_df = pd.read_pickle(term_path)
        terms = terms_df['terms'].values.flatten()

        seq = list(df['sequences'])
        label = list(df['prop_annotations'])
        assert len(seq) == len(label)
        return seq, label, terms


class ProteinSequenceDataset(Dataset):
    def __init__(self, sequence, targets, tokenizer, max_len):
        self.sequence = sequence
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        sequence = str(self.sequence[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'protein_sequence': sequence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


if __name__ == '__main__':

    pro_dataset = CustomProteinSequences(
        data_path='/Users/robin/xbiome/datasets/protein', split=True)
    for i in range(10):
        sample = pro_dataset[i]
        print(i, sample[0], sample[1])

    pro_dataset = ProtBertDataset(
        data_path='/Users/robin/xbiome/datasets/protein')
    print(pro_dataset.num_classes)
    for i in range(10):
        sample = pro_dataset[i]
        for key, val in sample.items():
            print(key, val)
