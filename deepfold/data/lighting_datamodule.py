import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .protein_dataset import ProtBertDataset


class LightingSeqenceDataModule(pl.LightningDataModule):
    """pytorch_lighting seqence Dataset Model."""
    def __init__(self,
                 data_path='dataset/',
                 tokenizer_name='Rostlab/prot_bert_bfd',
                 batch_size=16,
                 max_length=1024):
        super().__init__()

        self.data_path = data_path
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_token_len = max_length

    def setup(self, stage: str):
        self.train_dataset = ProtBertDataset(
            data_path=self.data_path,
            split='train',
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_token_len)

        self.val_dataset = ProtBertDataset(data_path=self.data_path,
                                           split='valid',
                                           tokenizer_name=self.tokenizer_name,
                                           max_length=self.max_token_len)
        self.test_dataset = ProtBertDataset(data_path=self.data_path,
                                            split='test',
                                            tokenizer_name=self.tokenizer_name,
                                            max_length=self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
