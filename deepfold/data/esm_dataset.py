from typing import Dict, List, Tuple

import esm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.constant import DEFAULT_ESM_MODEL, ESM_LIST


class ESMDataset(Dataset):
    def __init__(self, model_dir: str):
        super().__init__()
        if model_dir not in ESM_LIST:
            print(
                f"Model dir '{model_dir}' not recognized. Using '{DEFAULT_ESM_MODEL}' as default"
            )
            model_dir = DEFAULT_ESM_MODEL

        self._model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            model_dir)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.data, self.labels = self.data_to_tensor(data_df, terms_dict)

    def __len__(self):
        return self.len_df

    def __getitem__(self, idx):
        data = self.data[idx, :]
        label = self.labels[idx, :]
        return data, label

    def data_to_tensor(self, data_df, terms_dict):
        labels = np.zeros((len(data_df), len(terms_dict)), dtype=np.int32)
        esm_input = data_df.iloc[:, [1, 3]].values

        convert_labels, convert_strs, convert_tokens = self.batch_converter(
            esm_input)

        for i, row in enumerate(data_df.itertuples()):
            for t_id in row.prop_annotations:
                if t_id in terms_dict:
                    labels[i, terms_dict[t_id]] = 1

        labels = torch.from_numpy(labels).int()

        return convert_tokens, labels

    def process_sequences_and_tokens(
            self, sequences_list: List[str]) -> Dict[str, torch.Tensor]:
        """Function to transform tokens string to IDs; it depends on the model
        used."""
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
        return encoded_inputs

    def load_data(self, data_file, terms_file):
        data_df = pd.read_pickle(data_file)
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['terms'].values.flatten()
        return data_df, terms

    def load_esm_model(self):
        model, alphabet = torch.hub.load('facebookresearch/esm:main',
                                         self.esm_model)
        # 将未处理的 (labels + strings) batch 转化成(labels + tensor) batch
        batch_converter = alphabet.get_batch_converter()
        return batch_converter
