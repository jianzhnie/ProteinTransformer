import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NAMESPACES = {
    'cco': 'cellular_component',
    'mfo': 'molecular_function',
    'bpo': 'biological_process'
}


class MultiModalDataset(Dataset):
    """MultiModalDataset."""
    def __init__(self,
                 data_path: str = 'dataset/',
                 file_name: str = 'xxx.pkl',
                 namespace: str = 'bpo'):
        super().__init__()

        self.file_path = os.path.join(data_path, file_name)
        self.terms_path = os.path.join(
            data_path, 'onto_embeddings_mean_bert_embedding.pkl')

        self.namespace = NAMESPACES[namespace]
        self.embeddings, self.labels, self.terms, self.goterm_embedding = self.load_dataset(
            self.file_path, self.terms_path)

        self.goterm_embedding = torch.from_numpy(
            np.array(self.goterm_embedding, dtype=np.float32))
        self.terms_dict = {v: i for i, v in enumerate(self.terms)}
        self.num_classes = len(self.terms_dict)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        embedding = self.embeddings[idx]
        label_list = self.labels[idx]
        multilabel = [0] * self.num_classes
        for t_id in label_list:
            if t_id in self.terms_dict:
                label_idx = self.terms_dict[t_id]
                multilabel[label_idx] = 1

        embeddings = torch.from_numpy(np.array(embedding, dtype=np.float32))
        labels = torch.from_numpy(np.array(multilabel))
        encoded_inputs = {'embeddings': embeddings, 'labels': labels}
        return encoded_inputs

    def load_dataset(self, data_path, term_path):
        df = pd.read_pickle(data_path)
        terms_df = pd.read_pickle(term_path)
        terms = terms_df['term'][terms_df['namespace'] ==
                                 self.namespace].values.flatten()
        text_embeddings = terms_df['embeddings'][terms_df['namespace'] ==
                                                 self.namespace].tolist()

        embeddings = list(df['esm_embeddings'])
        label = list(df['prop_annotations'])
        assert len(embeddings) == len(label)
        assert len(text_embeddings) == len(terms)
        return embeddings, label, terms, text_embeddings


def process_dataset_(data_path, file_name, namespace):
    term_path = os.path.join(data_path, file_name)
    terms_df = pd.read_csv(term_path)
    name = NAMESPACES[namespace]
    terms = terms_df['term'][terms_df['namespace'] == name]
    df = terms.to_frame(name='terms')
    path = os.path.join(data_path, namespace + '_terms.pkl')
    df.to_pickle(path)
    return terms


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_root = '/home/niejianzheng/xbiome/datasets/protein/cafa3/process'
    # data_root = '/Users/robin/xbiome/datasets/protein'
    for key in NAMESPACES.keys():
        process_dataset_(data_path=data_root,
                         file_name='GotermText.csv',
                         namespace=key)
    pro_dataset = MultiModalDataset(
        data_path=data_root,
        file_name='cco_esm1b_t33_650M_UR50S_embeddings_mean_test.pkl',
        namespace='cco')
    print(pro_dataset.num_classes)
    data_loader = DataLoader(pro_dataset, batch_size=8)

    for index, batch in enumerate(data_loader):
        for key, val in batch.items():
            print(key, val.shape, val)
        if index > 3:
            break
