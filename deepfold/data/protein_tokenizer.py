"""Tools for protein features."""
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import List

IUPAC_CODES = OrderedDict([('Ala', 'A'), ('Asx', 'B'), ('Cys', 'C'),
                           ('Asp', 'D'), ('Glu', 'E'), ('Phe', 'F'),
                           ('Gly', 'G'), ('His', 'H'), ('Ile', 'I'),
                           ('Lys', 'K'), ('Leu', 'L'), ('Met', 'M'),
                           ('Asn', 'N'), ('Pro', 'P'), ('Gln', 'Q'),
                           ('Arg', 'R'), ('Ser', 'S'), ('Thr', 'T'),
                           ('Sec', 'U'), ('Val', 'V'), ('Trp', 'W'),
                           ('Xaa', 'X'), ('Tyr', 'Y'), ('Glx', 'Z')])

IUPAC_VOCAB = OrderedDict([
    ('<pad>', 0), ('<mask>', 1), ('<cls>', 2), ('<sep>', 3), ('<unk>', 4),
    ('A', 5), ('B', 6), ('C', 7), ('D', 8), ('E', 9), ('F', 10), ('G', 11),
    ('H', 12), ('I', 13), ('K', 14), ('L', 15), ('M', 16), ('N', 17),
    ('O', 18), ('P', 19), ('Q', 20), ('R', 21), ('S', 22), ('T', 23),
    ('U', 24), ('V', 25), ('W', 26), ('X', 27), ('Y', 28), ('Z', 29)
])

UNIREP_VOCAB = OrderedDict([('<pad>', 0), ('M', 1),
                            ('R', 2), ('H', 3), ('K', 4), ('D', 5), ('E', 6),
                            ('S', 7), ('T', 8), ('N', 9), ('Q', 10), ('C', 11),
                            ('U', 12), ('G', 13), ('P', 14), ('A', 15),
                            ('V', 16), ('I', 17), ('F', 18), ('Y', 19),
                            ('W', 20), ('L', 21), ('O', 22), ('X', 23),
                            ('Z', 23), ('B', 23), ('J', 23), ('<cls>', 24),
                            ('<sep>', 25)])


class Tokenizer(ABC):
    @abstractmethod
    def __init__(self):
        self.unknown_token_id = -1
        self.start_token = '<cls>'
        self.end_token = '<sep>'
        self.vocab = defaultdict(lambda: 'Not Present')
        self.tokens = list(self.vocab.keys())

    @abstractmethod
    def tokenize(self, sequence):
        pass

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # @property
    # def mask_token(self) -> str:
    #     if '<mask>' in self.vocab:
    #         return '<mask>'
    #     else:
    #         raise RuntimeError(' vocab does not support masking')

    def convert_token_to_id(self, token):
        """Converts a token to an id.

        Args:
            token: Token.

        Returns:
            id: The id of the input token.
        """
        if token not in self.vocab:
            return self.unknown_token_id
        else:
            return self.vocab[token]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert multiple tokens to ids.

        Args:
            tokens: The list of tokens.

        Returns:
            ids: The id list of the input tokens.
        """
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the
        vocab."""
        try:
            return self.tokens[index]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{index}'")

    def convert_ids_to_tokens(self, indices: List[int]) -> List[str]:
        return [self.convert_id_to_token(id_) for id_ in indices]

    @staticmethod
    def convert_tokens_to_string(tokens: str) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        return ''.join(tokens)

    def gen_token_ids(self, sequence):
        """Generate the list of token ids according the input sequence.

        Args:
            sequence: Sequence to be tokenized.

        Returns:
            token_ids: The list of token ids.
        """
        tokens = [self.start_token]
        tokens.extend(self.tokenize(sequence))
        tokens.append(self.end_token)
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids


class ProteinTokenizer(Tokenizer):
    """Protein Tokenizer."""
    def __init__(self):
        # "<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4
        super().__init__()

        self.padding_token = '<pad>'
        self.mask_token = '<mask>'
        self.start_token = self.class_token = '<cls>'
        self.end_token = self.separate_token = '<sep>'
        self.unknown_token = '<unk>'

        self.padding_token_id = 0
        self.mask_token_id = 1
        self.start_token_id = self.class_token_id = 2
        self.end_token_id = self.separate_token_id = 3
        self.unknown_token_id = 4

        self.vocab = OrderedDict([(self.padding_token, 0),
                                  (self.mask_token, 1), (self.class_token, 2),
                                  (self.separate_token, 3),
                                  (self.unknown_token, 4), ('A', 5), ('B', 6),
                                  ('C', 7), ('D', 8), ('E', 9), ('F', 10),
                                  ('G', 11), ('H', 12), ('I', 13), ('K', 14),
                                  ('L', 15), ('M', 16), ('N', 17), ('O', 18),
                                  ('P', 19), ('Q', 20), ('R', 21), ('S', 22),
                                  ('T', 23), ('U', 24), ('V', 25), ('W', 26),
                                  ('X', 27), ('Y', 28), ('Z', 29)])
        self.tokens = list(self.vocab.keys())

    def tokenize(self, sequence):
        """Split the sequence into token list.

        Args:
            sequence: The sequence to be tokenized.

        Returns:
            tokens: The token lists.
        """
        return [x for x in sequence]


if __name__ == '__main__':
    tokenizer = ProteinTokenizer()
    print(tokenizer.tokens)
    print(tokenizer.convert_id_to_token(5))
    print(tokenizer.convert_token_to_id('hangzhou'))
    print(tokenizer.vocab_size)
    print(tokenizer.gen_token_ids(['A', 'G']))
