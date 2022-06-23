import os

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .utils.onto_parser import OntologyParser


class OntoTextDataset(Dataset):
    def __init__(self,
                 data_dir,
                 tokenizer_name='Rostlab/prot_bert_bfd',
                 obo_file='data/go.obo',
                 max_length=1024):
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
            return_tensors='pt'  # PyTorch Tensor format
        )
        encode_input['labels'] = torch.tensor(label)

        return encode_input

    def collate_fn(self, examples):

        term_ids = torch.tensor([ex[0] for ex in examples])
        ancestor_ids = torch.tensor([ex[1] for ex in examples])
        namespace_ids = torch.tensor([ex[2] for ex in examples])

        encoded_inputs = {
            'term_ids': term_ids,
            'neighbor_ids': ancestor_ids,
            'labels': namespace_ids
        }
        return encoded_inputs

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
