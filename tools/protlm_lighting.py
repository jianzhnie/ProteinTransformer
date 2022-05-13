import sys

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import BertConfig

from deepfold.data.lighting_datamodule import LightingESMDataModule
from deepfold.models.esm_model import ESMTransformer
from deepfold.models.transformers.lighting_model import (
    BertForMultiLabelSequenceClassification, LightningTransformer)

sys.path.append('../')

if __name__ == '__main__':
    seed_everything(42)
    model_name = 'esm1b_t33_650M_UR50S'
    data_path = '/Users/robin/xbiome/datasets/protein'
    data_path = '/home/niejianzheng/xbiome/datasets/protein'
    checkpoint_dir = '/Users/robin/xbiome/work_dir'
    pretrain_model = 'esm1b'
    dm = LightingESMDataModule(data_path=data_path,
                               model_name='esm1b_t33_650M_UR50S',
                               batch_size=8,
                               max_length=1024)
    dm.setup('fit')
    num_classes = dm.train_dataset.num_classes
    if pretrain_model == 'protbert':
        model_config = BertConfig.from_pretrained(model_name,
                                                  num_labels=num_classes)
        transformer_model = BertForMultiLabelSequenceClassification.from_pretrained(
            model_name, config=model_config)
    else:
        transformer_model = ESMTransformer(model_dir='esm1b_t33_650M_UR50S',
                                           pool_mode='cls',
                                           num_labels=num_classes)

    lighting_model = LightningTransformer(
        model=transformer_model, num_labels=dm.train_dataset.num_classes)

    # 初始化ModelCheckpoint回调，并设置要监控的量。
    # monitor：需要监控的量，string类型。
    # 例如'val_loss'（在training_step() or validation_step()函数中通过self.log('val_loss', loss)进行标记）；
    # 默认为None，只保存最后一个epoch的模型参数,
    model_checkpoint = ModelCheckpoint(checkpoint_dir,
                                       filename=model_name,
                                       monitor='val_loss',
                                       save_top_k=5)
    early_stopping = EarlyStopping('val_loss')

    trainer = Trainer(
        devices=4,
        accelerator='gpu',
        strategy='deepspeed_stage_2',
        precision=16,
        amp_level='O2',
        accumulate_grad_batches=4,  # 每k次batches累计一次梯度
        auto_scale_batch_size=True,
        callbacks=[model_checkpoint, early_stopping],  # 添加回调函数或回调函数列表
        deterministic=True,
        max_epochs=30,  # 最多训练轮数
        min_epochs=5,
        fast_dev_run=True,
        log_every_n_steps=10)

    trainer.fit(lighting_model, datamodule=dm)
