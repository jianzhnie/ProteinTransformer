import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .esm_dataset import ESMDataset
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


class ESMDataModule(pl.LightningDataModule):
    """pytorch_lighting seqence Dataset Model."""
    def __init__(self,
                 data_path: str = 'dataset/',
                 model_name: str = None,
                 batch_size: int = 16,
                 max_length: int = 1024):
        super().__init__()

        self.data_path = data_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_token_len = max_length

    def setup(self, stage: str):
        self.train_dataset = ESMDataset(data_path=self.data_path,
                                        split='train',
                                        model_dir=self.model_name,
                                        max_length=self.max_token_len)

        self.valid_dataset = ESMDataset(data_path=self.data_path,
                                        split='valid',
                                        model_dir=self.model_name,
                                        max_length=self.max_token_len)
        self.test_dataset = ESMDataset(data_path=self.data_path,
                                       split='test',
                                       model_dir=self.model_name,
                                       max_length=self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          pin_memory=True)
