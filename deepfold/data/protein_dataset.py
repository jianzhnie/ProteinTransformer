import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .protein_tokenizer import ProteinTokenizer


class ProteinSequenceDataset(Dataset):

    def __init__(self, sequence, targets, tokenizer, max_len):
        self.sequence = sequence
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        sequence = str(self.sequence[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'protein_sequence': sequence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class ProteinSequences(Dataset):

    def __init__(self, data_file, terms_file):
        super().__init__()

        self.data_df, terms = self.load_data(data_file, terms_file)
        # convert terms to dict
        self.terms_dict = {v: i for i, v in enumerate(terms)}

        # self.
        self.terms = terms
        self.num_classes = len(terms)
        self.tokenizer = ProteinTokenizer()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        seqence = self.data_df['sequences'].iloc[idx]
        label_list = self.data_df['prop_annotations'].iloc[idx]

        label = [0] * self.num_classes
        for t_id in label_list:
            if t_id in self.terms_dict:
                label_idx = self.terms_dict[t_id]
                label[label_idx] = 1

        token_ids = self.tokenizer.gen_token_ids(seqence)
        return token_ids, label

    def collate_fn(self, examples):
        # 从独立样本集合中构建batch输入输出
        inputs = [torch.tensor(ex[0]) for ex in examples]
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.float)
        # 对batch内的样本进行padding，使其具有相同长度
        inputs = pad_sequence(inputs,
                              batch_first=True,
                              padding_value=self.tokenizer.padding_token_id)

        return (inputs, targets)

    def load_data(self, data_file, terms_file):
        data_df = pd.read_pickle(data_file)
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['terms'].values.flatten()
        return data_df, terms


if __name__ == '__main__':

    pro_dataset = ProteinSequences(
        data_file='/Users/robin/xbiome/datasets/protein/train_data.pkl',
        terms_file='/Users/robin/xbiome/datasets/protein/terms.pkl')
    for i in range(10):
        sample = pro_dataset[i]
        print(i, sample['seq'], sample['label'])
