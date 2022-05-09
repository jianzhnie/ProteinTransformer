import sys

import torch
from pytorch_lightning import Trainer, seed_everything

from deepfold.data.lighting_datamodule import LightingSeqenceDataModule
from deepfold.models.transformers.lighting_model import LightningTransformer

sys.path.append('../')

if __name__ == '__main__':
    seed_everything(42)
    data_path = '/Users/robin/xbiome/datasets/protein'
    dm = LightingSeqenceDataModule(data_path=data_path,
                                   tokenizer_name='Rostlab/prot_bert_bfd',
                                   batch_size=16,
                                   max_length=1024)
    dm.setup('fit')
    model = LightningTransformer(model_name_or_path='Rostlab/prot_bert_bfd',
                                 num_labels=dm.train_dataset.num_classes)

    trainer = Trainer(
        max_epochs=1,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        # limiting got iPython runs
    )
    trainer.fit(model, datamodule=dm)
