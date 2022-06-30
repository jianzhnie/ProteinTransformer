import os
import random
from typing import Dict

import pandas as pd
import torch
from allennlp.modules.elmo import batch_to_ids
from torch.utils.data import Dataset

from deepfold.data.utils.ontology import Ontology


class Seq2VecDataset(Dataset):
    """ESMDataset."""
    def __init__(self,
                 label_map,
                 namespace: str = 'bpo',
                 root_path: str = 'dataset/',
                 file_name: str = 'xxx.pkl',
                 max_length: int = 1024,
                 truncate: bool = False,
                 random_crop: bool = False):
        super().__init__()

        sub_path = os.path.join(root_path, namespace)
        self.data_path = os.path.join(sub_path, file_name)
        self.seqs, self.labels = self.load_dataset(self.data_path)

        self.terms_dict = label_map
        self.num_classes = len(self.terms_dict)
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

        label_list = self.labels[idx]
        multilabel = [0] * self.num_classes
        for t_id in label_list:
            if t_id in self.terms_dict:
                label_idx = self.terms_dict[t_id]
                multilabel[label_idx] = 1
        return sequence, multilabel

    def load_dataset(self, data_path):
        df = pd.read_pickle(data_path)
        seq = list(df['sequences'])
        label = list(df['annotations'])
        assert len(seq) == len(label)
        return seq, label

    def collate_fn(self, examples) -> Dict[str, torch.Tensor]:
        """Function to transform tokens string to IDs; it depends on the model
        used."""
        sequences_list = [ex[0] for ex in examples]
        multilabel_list = [ex[1] for ex in examples]

        all_tokens = batch_to_ids(sequences_list)

        if self.truncate:
            all_tokens = all_tokens[:, :self.max_length]

        all_tokens = all_tokens.to('cpu')
        encoded_inputs = {
            'inputs': all_tokens,
        }
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
    data_path = '/data/xbiome/protein_classification/cafa3'
    go_file = os.path.join(data_path, 'go_cafa3.obo')
    bpo_path = os.path.join(data_path, 'bpo')
    bpo_train_data_file = os.path.join(bpo_path, 'bpo_train_data.pkl')
    bpo_test_data_file = os.path.join(bpo_path, 'bpo_test_data.pkl')
    # constract label map dict
    go_ont = Ontology(go_file, with_rels=True)
    bpo_label_map = {
        v: i
        for i, v in enumerate(go_ont.get_namespace_terms('biological_process'))
    }
    bpo_train_data = pd.read_pickle(bpo_train_data_file)
    bpo_train_dataset = Seq2VecDataset(label_map=bpo_label_map,
                                       namespace='bpo',
                                       root_path=data_path,
                                       file_name='bpo_train_data.pkl')
    from torch.utils.data import DataLoader
    data_loader = DataLoader(bpo_train_dataset,
                             batch_size=8,
                             collate_fn=bpo_train_dataset.collate_fn)

    for index, batch in enumerate(data_loader):
        for key, val in batch.items():
            print(key, val.shape)
        if index > 10:
            break
