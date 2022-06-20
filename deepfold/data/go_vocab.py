import os

from torch.utils.data import Dataset

from .utils.onto_parser import (BIOLOGICAL_PROCESS, CELLULAR_COMPONENT,
                                MOLECULAR_FUNCTION)


class GoVocab(object):
    """Vocabulary for Go Terms."""
    def __init__(self, go_terms):
        self.ont = go_terms
        # add GO terms in the tokenizer
        self.terms = sorted(list(set(go_terms.keys())))
        self.vocab_sz = len(self.terms)
        # zero is padding
        self.term2index = dict([(term, i)
                                for i, term in enumerate(self.terms)])
        self.index2term = dict([(i, term)
                                for i, term in enumerate(self.terms)])

    def __len__(self):
        return len(self.term2index)

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.index2term[int(index)] for index in indices]
        return self.index2term[indices]

    def to_ids(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.term2index(token) for token in tokens]
        return self.term2index.get(tokens, self.unk)


def save_vocab(vocab, path):
    with open(path, 'w') as writer:
        writer.write('\n'.join(vocab.index2term))


def read_vocab(path):
    with open(path, 'r') as f:
        tokens = f.read().split('\n')
    return GoVocab(tokens)


class OntoDataset(Dataset):
    def __init__(self, vocab):
        self._build_dataset()

    def build_dataset(self, path):
        """Create a train dataset from obo file."""
        dat_fin = os.path.join(path, 'ds.tsv')

        name2code = {
            'biological_process': 0,
            'cellular_component': 1,
            'molecular_function': 2
        }

        # create dataset
        with open(dat_fin, 'w') as f:
            # loop over GO terms
            for t in self.vocab.ont:
                # skip roots
                if t == BIOLOGICAL_PROCESS or t == CELLULAR_COMPONENT or t == MOLECULAR_FUNCTION:
                    continue
                namespace = self.ont[t]['namespace']
                ancestors = self.ont.get_ancestors(t)
                ancestors = set([t for a in ancestors for t in a])

                datapoint = '{}\t{}\t{}\n'.format(t,
                                                  '!'.join(sorted(ancestors)),
                                                  name2code[namespace])

                f.write(datapoint)

        # generate dataset
        self.data_fin = dat_fin
