import numpy as np
import pandas as pd
import torch
from protein_tokenizer import ProteinTokenizer
from torch.utils.data import Dataset


class ProteinSequences(Dataset):
    def __init__(self, data_file, terms_file):
        super().__init__()

        self.data_df, terms = self.load_data(data_file, terms_file)
        # convert terms to dict
        self.terms_dict = {v: i for i, v in enumerate(terms)}

        # self.
        self.terms = terms
        self.num_classes = len(terms)
        self.tokenizer = ProteinTokenizer()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        seqence = self.data_df['sequences'].iloc[idx]
        label_list = self.data_df['prop_annotations'].iloc[idx]

        label = np.zeros(self.num_classes, dtype=np.int8)
        for t_id in label_list:
            if t_id in self.terms_dict:
                label_idx = self.terms_dict[t_id]
                label[label_idx] = 1

        token_ids = self.tokenizer.gen_token_ids(seqence)
        token_ids = np.array(token_ids)
        input_ids = torch.from_numpy(token_ids)
        label = torch.from_numpy(label)

        return {'seq': input_ids, 'label': label}

    def load_data(self, data_file, terms_file):
        data_df = pd.read_pickle(data_file)
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['terms'].values.flatten()
        return data_df, terms


if __name__ == '__main__':

    pro_dataset = ProteinSequences(
        data_file='/Users/robin/xbiome/datasets/protein/train_data.pkl',
        terms_file='/Users/robin/xbiome/datasets/protein/terms.pkl')
    for i in range(10):
        sample = pro_dataset[i]
        print(i, sample['seq'], sample['label'])
