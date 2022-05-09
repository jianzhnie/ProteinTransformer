import sys

import torch
from pytorch_lightning import Trainer, seed_everything
from transformers import BertConfig

from deepfold.data.lighting_datamodule import LightingSeqenceDataModule
from deepfold.models.transformers.lighting_model import (
    BertForMultiLabelSequenceClassification, LightningTransformer)

sys.path.append('../')

if __name__ == '__main__':
    seed_everything(42)
    model_name = 'Rostlab/prot_bert_bfd'
    data_path = '/Users/robin/xbiome/datasets/protein'
    dm = LightingSeqenceDataModule(data_path=data_path,
                                   tokenizer_name='Rostlab/prot_bert_bfd',
                                   batch_size=1,
                                   max_length=64)
    dm.setup('fit')

    num_classes = dm.train_dataset.num_classes
    model_config = BertConfig.from_pretrained(model_name,
                                              num_labels=num_classes)
    transformer_model = BertForMultiLabelSequenceClassification.from_pretrained(
        model_name, config=model_config)
    print(type(transformer_model))
    lighting_model = LightningTransformer(
        model=transformer_model, num_labels=dm.train_dataset.num_classes)

    trainer = Trainer(
        max_epochs=1,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        # limiting got iPython runs
    )
    trainer.fit(lighting_model, datamodule=dm)
