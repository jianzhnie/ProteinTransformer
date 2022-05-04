import numpy as np

MAXLEN = 2000


class AminoacidsVocab(object):
    def __init__(self, maxlen=2000):
        self.acids_vocab = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M',
            'F', 'P', 'S', 'T', 'W', 'Y', 'V'
        ]
        self.invalid_acids = set(['U', 'O', 'B', 'Z', 'J', 'X', '*'])
        self.acids_num = len(self.acids_vocab)
        self.maxlen = maxlen

        # The list of unique tokens
        self.token_to_idx = {
            token: idx + 1
            for idx, token in enumerate(self.acids_vocab)
        }

        self.idx_to_token = {
            idx: token
            for token, idx in self.token_to_idx.items()
        }

        self.acids_ngram = self.gen_acids_ngram()

    def gen_acids_ngram(self):
        acids_ngram = {}
        for i in range(20):
            for j in range(20):
                for k in range(20):
                    ngram = self.acids_vocab[i] + self.acids_vocab[
                        j] + self.acids_vocab[k]
                    index = 400 * i + 20 * j + k + 1
                    acids_ngram[ngram] = index

        return acids_ngram

    def is_ok(self, seq):
        for c in seq:
            if c in self.invalid_acids:
                return False
        return True

    def to_ngrams(self, seq):
        l = min(self.maxlen, len(seq) - 3)
        ngrams = np.zeros((l, ), dtype=np.int32)
        for i in range(l):
            ngrams[i] = self.acids_ngram.get(seq[i:i + 3], 0)
        return ngrams

    def to_onehot(self, seq, start=0):
        # range(1, 21) 代表20种氨基酸
        onehot = np.zeros((self.maxlen, 21), dtype=np.int32)
        l = min(self.maxlen, len(seq))
        for i in range(start, start + l):
            onehot[i, self.token_to_idx.get(seq[i - start], 0)] = 1
        # index==0 的位置 表示 Unknow
        onehot[0:start, 0] = 1
        onehot[start + l:, 0] = 1
        return onehot


if __name__ == '__main__':
    vocab = AminoacidsVocab()
    print(vocab.token_to_idx)
    seq = 'MASNTVSAQGGSNRPVRDFSNIQDVAQFLLFDPIWNEQPGSIVPWKMNREQAL\
        AERYPELQTSEPSEDYSGPVESLELLPLEIKLDIMQYLSWEQISWCKHPWLWTRWYK\
        DNVVRVSAITFEDFQREYAFPEKIQEIHFTDTRAEEIKAILETTPNVTRLVIRRIDDMNY\
        NTHGDLGLDDLEFLTHLMVEDACGFTDFWAPSLTHLTIKNLDMHPRWFGPVMDGIKSMQST\
        LKYLYIFETYGVNKPFVQWCTDNIETFYCTNSYRYENVPRPIYVWVLFQEDEWHGYRVEDNK\
        FHRRYMYSTILHKRDTDWVENNPLKTPAQVEMYKFLLRISQLNRDGTGYESDSDPENEHFDDE\
        SFSSGEEDSSDEDDPTWAPDSDDSDWETETEEEPSVAARILEKGKLTITNLMKSLGFKPKPKKIQ\
        SIDRYFCSLDSNYNSEDEDFEYDSDSEDDDSDSEDDC'

    onehot = vocab.to_onehot(seq)
    print(len(seq))
    print(onehot)
    print(onehot.shape)
