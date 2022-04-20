import esm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ESMDataset(Dataset):

    def __init__(self, data_file, terms_file, esm_model='esm1_t6_43M_UR50S'):
        super().__init__()

        data_df, terms = self.load_data(data_file, terms_file)
        # convert terms to dict
        terms_dict = {v: i for i, v in enumerate(terms)}
        self.len_df = len(data_df)
        self.nb_classes = len(terms)
        self.ems_model = esm_model
        self.batch_converter = self.load_esm_model()
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
        convert_tokens = convert_tokens[:, :1024]

        for i, row in enumerate(data_df.itertuples()):
            for t_id in row.prop_annotations:
                if t_id in terms_dict:
                    labels[i, terms_dict[t_id]] = 1

        labels = torch.from_numpy(labels).int()

        return convert_tokens, labels

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
