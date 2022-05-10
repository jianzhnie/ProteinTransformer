from typing import Dict
import os
import esm
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.constant import DEFAULT_ESM_MODEL, ESM_LIST


class ESMDataset(Dataset):

    def __init__(self,
                 data_path: str = 'dataset/',
                 split: str = 'train',
                 model_dir: str = None):
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

        if model_dir not in ESM_LIST:
            print(
                f"Model dir '{model_dir}' not recognized. Using '{DEFAULT_ESM_MODEL}' as default"
            )
            model_dir = DEFAULT_ESM_MODEL

        self._model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            model_dir)
        self.batch_converter = self.alphabet.get_batch_converter()

    def __len__(self):
        return self.labels

    def __getitem__(self, idx):

        seq = self.serqs[idx]
        label_list = self.labels[idx]
        multilabel = [0] * self.num_classes
        for t_id in label_list:
            if t_id in self.terms_dict:
                label_idx = self.terms_dict[t_id]
                multilabel[label_idx] = 1

        return seq, multilabel

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
        multilabel_list = [ex[1] for ex in examples]

        if self.is_msa:
            _, _, all_tokens = self.batch_converter(sequences_list)
        else:
            _, _, all_tokens = self.batch_converter([
                ('', sequence) for sequence in sequences_list
            ])

        all_tokens = all_tokens.to('cpu')
        encoded_inputs = {
            'input_ids': all_tokens,
            'attention_mask':
            1 * (all_tokens != self.token_to_id(self.pad_token)),
            'token_type_ids': torch.zeros(all_tokens.shape),
        }
        encoded_inputs['labels'] = torch.tensor(multilabel_list)
        return encoded_inputs