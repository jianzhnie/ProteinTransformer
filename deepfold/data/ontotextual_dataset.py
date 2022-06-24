import os

import pandas as pd
from deepfold import data
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .utils.onto_parser import OntologyParser


class OntoTextDataset(Dataset):
    def __init__(self,
                 data_dir,
                 tokenizer_name='Rostlab/prot_bert_bfd',
                 obo_file=None,
                 max_length=512):
        
        if obo_file is None:
            obo_file = os.path.join(data_dir, 'go.obo')

        self.ontparser = OntologyParser(obo_file,
                                        with_rels=True,
                                        include_alt_ids=False)
        self.ont = self.ontparser.ont
        self.all_terms = sorted(list(set(self.ont.keys())))
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                       do_lower_case=False)
        self.name2code = {
            'biological_process': 0,
            'cellular_component': 1,
            'molecular_function': 2
        }
        self.data = self.build_dataset()

        # save data
        data_fin = os.path.join(data_dir, 'GotermText.txt')
        if os.path.exists(data_fin):
            pass
        else:
            self.save_processed_data(data_fin)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        term, name, namespace, textdef = self.data[idx]
        label = self.name2code[namespace]

        encode_input = self.tokenizer(
            textdef,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,  # Truncate data beyond max length
        )
        encode_input = {key: torch.tensor(val) for key, val in encode_input.items()}

        return encode_input

    def build_dataset(self):
        """Create a train dataset from obo file."""
        data = []
        # loop over GO terms
        for term in self.all_terms:
            namespace = self.ont[term]['namespace']
            name = self.ont[term]['name']
            textdef = self.ont[term]['def']
            data.append([term, name, namespace, textdef])
        return data

    def save_processed_data(self, data_fin):
        # create dataset
        with open(data_fin, 'w+') as f:
            # loop over GO terms
            for term in self.all_terms:
                # skip roots
                namespace = self.ont[term]['namespace']
                name = self.ont[term]['name']
                textdef = self.ont[term]['def']
                datapoint = '{}\t{}\t{}\t{}\n'.format(term, name, namespace,
                                                      textdef)

                f.write(datapoint)

        df = pd.DataFrame(self.data,
                          columns=['term', 'name', 'namespace', 'deftext'])
        data_fin = data_fin.replace('.txt', '.csv')
        df.to_csv(data_fin, index=False)
