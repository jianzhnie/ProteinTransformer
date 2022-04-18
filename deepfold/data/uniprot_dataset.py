import numpy as np
import torch
from torch.utils.data import Dataset

from .aminoacids import MAXLEN, AminoacidsVocab


# ------------------------------------------------------------------------------------------
# Customized pytorch Dateset for annotated sequences
class AnnotatedSequences(Dataset):
    def __init__(self,
                 data_frame,
                 terms,
                 transform=None,
                 target_transform=None,
                 data_type='one-hot'):
        super().__init__()
        # convert terms to dict
        terms_dict = {v: i for i, v in enumerate(terms)}
        # convert to tensor
        self.aminoacids_vocab = AminoacidsVocab()
        if data_type in ['one-hot', 'One-hot']:
            data_tensor, labels = self.df_to_tensor(data_frame, len(terms),
                                                    terms_dict)
        if data_type in ['label-index', 'Label-index']:
            data_tensor, labels = self.df_to_tensor_label_index(
                data_frame, len(terms), terms_dict)
        # self.
        self.data_type = data_type
        self.terms = terms
        self.nb_classes = len(terms)
        self.labels = labels
        self.data = data_tensor
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if self.data_type in ['one-hot', 'One-hot']:
            return len(self.data)
        if self.data_type in ['label-index', 'Label-index']:
            return self.data.size()[1]

    def __getitem__(self, idx):
        if self.data_type in ['one-hot', 'One-hot']:
            data = self.data[idx]
            label = self.labels[idx]
        if self.data_type in ['label-index', 'Label-index']:
            data = self.data[:, idx]
            label = self.labels[:, idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

    def df_to_tensor(self, df, nb_classes, terms_dict):
        # [batch, seq_len, num_aa_feature]
        data_onehot = np.zeros((len(df), MAXLEN, 21), dtype=np.float32)
        # [batch, num_classes]
        labels = np.zeros((len(df), nb_classes), dtype=np.int32)

        for i, row in enumerate(df.itertuples()):
            seq = row.sequences
            onehot = self.aminoacids_vocab.to_onehot(seq)
            data_onehot[i, :, :] = onehot
            for t_id in row.prop_annotations:
                if t_id in terms_dict:
                    labels[i, terms_dict[t_id]] = 1

        data_onehot = torch.from_numpy(data_onehot).float()
        labels = torch.from_numpy(labels).int()
        return data_onehot, labels

    # data type for transformer
    def df_to_tensor_label_index(self, df, nb_classes, terms_dict):
        # [seq_len, batch]
        data_index = np.zeros((MAXLEN, len(df)), dtype=np.float32)
        # [num_classes, batch]
        labels = np.zeros((nb_classes, len(df)), dtype=np.int32)

        for i, row in enumerate(df.itertuples()):
            seq = row.sequences
            label_index = self.aminoacids_vocab.to_label_index(seq)
            data_index[:, i] = label_index
            for t_id in row.prop_annotations:
                if t_id in terms_dict:
                    labels[terms_dict[t_id], i] = 1

        data_index = torch.from_numpy(data_index).int()
        labels = torch.from_numpy(labels).int()
        return data_index, labels


# Customized pytorch Dateset for annotated sequences of arbitrary length
class AnnotatedSequencesXL(Dataset):
    def __init__(self,
                 data_frame,
                 terms,
                 transform=None,
                 target_transform=None,
                 data_type='one-hot'):
        super().__init__()
        # convert terms to dict
        terms_dict = {v: i for i, v in enumerate(terms)}
        # convert to tensor
        if data_type in ['one-hot', 'One-hot']:
            data_tensor, labels = self.df_to_tensor(data_frame, len(terms),
                                                    terms_dict)
        if data_type in ['label-index', 'Label-index']:
            data_tensor, labels = self.df_to_tensor_label_index(
                data_frame, len(terms), terms_dict)
        # self.
        self.vocab = AminoacidsVocab()
        self.data_type = data_type
        self.terms = terms
        self.nb_classes = len(terms)
        self.labels = labels
        self.data = data_tensor
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if self.data_type in ['one-hot', 'One-hot']:
            return len(self.data)
        if self.data_type in ['label-index', 'Label-index']:
            return self.data.size()[1]

    def __getitem__(self, idx):
        if self.data_type in ['one-hot', 'One-hot']:
            data = self.data[idx]
            label = self.labels[idx]
        if self.data_type in ['label-index', 'Label-index']:
            data = self.data[:, idx]
            label = self.labels[:, idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

    def df_to_tensor(self, df, nb_classes, terms_dict):
        # [batch, num_aa_feature, seq_len]
        data_onehot = np.zeros((len(df), 21, MAXLEN), dtype=np.float32)
        # [batch, num_classes]
        labels = np.zeros((len(df), nb_classes), dtype=np.int32)

        for i, row in enumerate(df.itertuples()):
            seq = row.sequences
            onehot = self.vocab.to_onehot(seq)
            data_onehot[i, :, :] = onehot
            for t_id in row.prop_annotations:
                if t_id in terms_dict:
                    labels[i, terms_dict[t_id]] = 1

        data_onehot = torch.from_numpy(data_onehot).float()
        labels = torch.from_numpy(labels).int()
        return data_onehot, labels

    # data type for transformer
    def df_to_tensor_label_index(self, df, nb_classes, terms_dict):
        # [seq_len, batch]
        data_index = np.zeros((MAXLEN, len(df)), dtype=np.float32)
        # [num_classes, batch]
        labels = np.zeros((nb_classes, len(df)), dtype=np.int32)

        for i, row in enumerate(df.itertuples()):
            seq = row.sequences
            label_index = self.vocab.to_label_index(seq)
            data_index[:, i] = label_index
            for t_id in row.prop_annotations:
                if t_id in terms_dict:
                    labels[terms_dict[t_id], i] = 1

        data_index = torch.from_numpy(data_index).int()
        labels = torch.from_numpy(labels).int()
        return data_index, labels
