import os
import random
from typing import Dict

import pandas as pd
import torch
from allennlp.modules.elmo import batch_to_ids
from torch.utils.data import Dataset


class Seq2VecDataset(Dataset):
    """ESMDataset."""
    def __init__(self,
                 data_path: str = 'dataset/',
                 file_name: str = 'xxx.pkl',
                 max_length: int = 1024,
                 truncate: bool = False,
                 random_crop: bool = False):
        super().__init__()

        self.file_path = os.path.join(data_path, file_name)
        self.terms_path = os.path.join(data_path, 'terms.pkl')

        self.seqs, self.labels, self.terms = self.load_dataset(
            self.file_path, self.terms_path)

        self.terms_dict = {v: i for i, v in enumerate(self.terms)}
        self.num_classes = len(self.terms)
        self.max_length = max_length
        self.truncate = truncate
        self.random_crop = random_crop

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sequence = self.seqs[idx]
        if self.random_crop:
            sequence = crop_sequence(sequence, crop_length=self.max_length - 2)
        if self.truncate:
            sequence = sequence[:self.max_length - 2]

        length = len(sequence)
        label_list = self.labels[idx]
        multilabel = [0] * self.num_classes
        for t_id in label_list:
            if t_id in self.terms_dict:
                label_idx = self.terms_dict[t_id]
                multilabel[label_idx] = 1
        return sequence, length, multilabel

    def load_dataset(self, data_path, term_path):
        df = pd.read_pickle(data_path)
        terms_df = pd.read_pickle(term_path)
        terms = terms_df['terms'].values.flatten()

        seq = list(df['sequences'])
        label = list(df['prop_annotations'])
        assert len(seq) == len(label)
        return seq, label, terms

    def collate_fn(self, examples) -> Dict[str, torch.Tensor]:
        """Function to transform tokens string to IDs; it depends on the model
        used."""
        sequences_list = [ex[0] for ex in examples]
        lengths = [ex[1] for ex in examples]
        multilabel_list = [ex[2] for ex in examples]

        all_tokens = batch_to_ids(sequences_list)

        if self.truncate:
            all_tokens = all_tokens[:, :self.max_length]

        all_tokens = all_tokens.to('cpu')
        encoded_inputs = {
            'inputs': all_tokens,
        }
        encoded_inputs['lengths'] = torch.tensor(lengths, dtype=torch.int)
        encoded_inputs['labels'] = torch.tensor(multilabel_list,
                                                dtype=torch.int)
        return encoded_inputs


def crop_sequence(sequence: str, crop_length: int) -> str:
    """If the length of the sequence is superior to crop_length, crop randomly
    the sequence to get the proper length."""
    if len(sequence) <= crop_length:
        return sequence
    else:
        start_idx = random.randint(0, len(sequence) - crop_length)
        return sequence[start_idx:(start_idx + crop_length)]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_root = '/home/niejianzheng/xbiome/datasets/protein'
    data_root = '/Users/robin/xbiome/datasets/protein'
    pro_dataset = Seq2VecDataset(data_path=data_root,
                                 file_name='test_data.pkl')
    print(pro_dataset.num_classes)
    data_loader = DataLoader(pro_dataset,
                             batch_size=8,
                             collate_fn=pro_dataset.collate_fn)

    for index, batch in enumerate(data_loader):
        for key, val in batch.items():
            print(key, val.shape)
        if index > 10:
            break
