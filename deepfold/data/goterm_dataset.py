import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils.onto_parser import (BIOLOGICAL_PROCESS, CELLULAR_COMPONENT,
                                MOLECULAR_FUNCTION, OntologyParser)


class GoVocab(object):
    """Vocabulary for Go Terms."""
    def __init__(self, terms):
        # add GO terms in the tokenizer
        self.terms = terms
        self.vocab_sz = len(self.terms)
        self.term2index = dict([(term, i)
                                for i, term in enumerate(self.terms)])
        self.index2term = dict([(i, term)
                                for i, term in enumerate(self.terms)])

    def __len__(self):
        return len(self.terms)

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.index2term[int(index)] for index in indices]
        return self.index2term[indices]

    def to_ids(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.term2index[token] for token in tokens]
        return self.term2index.get(tokens)


def save_vocab(vocab, path):
    with open(path, 'w') as writer:
        writer.write('\n'.join(vocab.index2term))


def read_vocab(path):
    with open(path, 'r') as f:
        tokens = f.read().split('\n')
    return GoVocab(tokens)


class OntoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        obo_file='data/go.obo',
    ):
        self.ontparser = OntologyParser(obo_file,
                                        with_rels=True,
                                        include_alt_ids=False)
        self.ont = self.ontparser.ont
        self.all_terms = sorted(list(set(self.ont.keys())))
        self.vocab = GoVocab(self.all_terms)

        self.root_terms = [
            BIOLOGICAL_PROCESS, MOLECULAR_FUNCTION, CELLULAR_COMPONENT
        ]
        self.filter_term = list(
            set(self.all_terms).difference(set(self.root_terms)))

        self.name2code = {
            'biological_process': 0,
            'cellular_component': 1,
            'molecular_function': 2
        }
        self.build_dataset(data_dir)

    def __len__(self):
        return len(self.filter_term)

    def __getitem__(self, idx):
        term = self.filter_term[idx]
        namespace = self.ont[term]['namespace']
        ancestors = self.ontparser.get_ancestors(term)
        ancestors = sorted(set([term for anc in ancestors for term in anc]))

        term_id = self.vocab.to_ids(term)
        neighbors_id = self.vocab.to_ids(ancestors)
        label_id = self.name2code[namespace]

        term_id = torch.tensor(term_id)
        neighbors_id = torch.tensor(neighbors_id)
        label_id = torch.tensor(label_id)

        term2onehot = F.one_hot(term_id, num_classes=self.vocab.vocab_sz)
        neighbors2onehot = F.one_hot(
            neighbors_id,
            num_classes=self.vocab.vocab_sz) * (1.0 / neighbors_id.shape[0])
        neighbors = torch.sum(neighbors2onehot, dim=0)
        labels = F.one_hot(label_id, num_classes=3)
        return term2onehot, neighbors, labels

    def load_dataset(self, path):
        with open(path) as f:
            for line in f.readlines():
                term, neighbors, namespace = line.strip().split('\t')

                term_id = self.vocab.to_ids(term)
                neighbors_id = [
                    self.vocab.to_ids(t) for t in neighbors.split(',')
                ]
                namespace = self.name2code(namespace)
                print(term_id, neighbors_id, namespace)

    def build_dataset(self, path):
        """Create a train dataset from obo file."""
        data_fin = os.path.join(path, 'ds.txt')
        # create dataset
        with open(data_fin, 'w+') as f:
            # loop over GO terms
            for t in self.all_terms:
                # skip roots
                if t in self.root_terms:
                    continue
                namespace = self.ont[t]['namespace']
                ancestors = self.ontparser.get_ancestors(t)
                ancestors = list(
                    set([term for anc in ancestors for term in anc]))

                datapoint = '{}\t{}\t{}\n'.format(t,
                                                  ','.join(sorted(ancestors)),
                                                  namespace)

                f.write(datapoint)
