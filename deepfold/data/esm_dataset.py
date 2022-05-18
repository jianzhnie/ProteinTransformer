import os
import random
import sys
from typing import Dict, List

import esm
import pandas as pd
import torch
from torch.utils.data import Dataset

from deepfold.utils.constant import DEFAULT_ESM_MODEL, ESM_LIST

sys.path.append('../../')


class ESMDataset(Dataset):
    def __init__(self,
                 data_path: str = 'dataset/',
                 split: str = 'train',
                 model_dir: str = None,
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

        self.terms_dict = {v: i for i, v in enumerate(self.terms)}
        self.num_classes = len(self.terms)
        self.max_length = max_length
        self.truncate = truncate
        self.random_crop = random_crop

        if model_dir not in ESM_LIST:
            print(
                f"Model dir '{model_dir}' not recognized. Using '{DEFAULT_ESM_MODEL}' as default"
            )
            model_dir = DEFAULT_ESM_MODEL

        self.is_msa = 'msa' in model_dir

        self._model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            model_dir)
        self.batch_converter = self.alphabet.get_batch_converter()

    @property
    def model(self) -> torch.nn.Module:
        """Return torch model."""
        return self._model

    @property
    def model_vocabulary(self) -> List[str]:
        """Returns the whole vocabulary list."""
        return list(self.alphabet.tok_to_idx.keys())

    @property
    def vocab_size(self) -> int:
        """Returns the whole vocabulary size."""
        return len(list(self.alphabet.tok_to_idx.keys()))

    @property
    def mask_token(self) -> str:
        """Representation of the mask token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.mask_idx]  # "<mask>"

    @property
    def pad_token(self) -> str:
        """Representation of the pad token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.padding_idx]  # "<pad>"

    @property
    def begin_token(self) -> str:
        """Representation of the beginning of sentence token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.cls_idx]  # "<cls>"

    @property
    def end_token(self) -> str:
        """Representation of the end of sentence token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.eos_idx]  # "<eos>"

    @property
    def does_end_token_exist(self) -> bool:
        """Returns true if a end of sequence token exists."""
        return self.alphabet.append_eos

    @property
    def token_to_id(self):
        """Returns a function which maps tokens to IDs."""
        return lambda x: self.alphabet.tok_to_idx[x]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sequence = self.seqs[idx]
        if self.random_crop:
            sequence = crop_sequence(sequence, crop_length=self.max_length - 2)

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

        if self.is_msa:
            labels, strs, all_tokens = self.batch_converter(sequences_list)
        else:
            labels, strs, all_tokens = self.batch_converter([
                ('', sequence) for sequence in sequences_list
            ])

        if self.truncate:
            all_tokens = all_tokens[:, :self.max_length - 2]

        all_tokens = all_tokens.to('cpu')
        encoded_inputs = {
            'input_ids': all_tokens,
            # 'attention_mask':
            # 1 * (all_tokens != self.token_to_id(self.pad_token)),
            # 'token_type_ids': torch.zeros(all_tokens.shape),
        }
        encoded_inputs['lengths'] = torch.tensor(lengths)
        encoded_inputs['labels'] = torch.tensor(multilabel_list)
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
    # data_root = '/Users/robin/xbiome/datasets/protein'
    pro_dataset = ESMDataset(data_path=data_root,
                             model_dir='esm1b_t33_650M_UR50S')
    print(pro_dataset.num_classes)
    data_loader = DataLoader(pro_dataset,
                             batch_size=8,
                             collate_fn=pro_dataset.collate_fn)

    for index, batch in enumerate(data_loader):
        for key, val in batch.items():
            print(key, val.shape, val)
        if index > 10:
            break
