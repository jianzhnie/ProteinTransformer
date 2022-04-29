import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from deepfold.data.protein_dataset import ProtBertDataset


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_name)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == '__main__':
    model_name = 'Rostlab/prot_bert_bfd'
    data_root = '/Users/robin/xbiome/datasets/protein'
    train_dataset = ProtBertDataset(
        data_path=data_root,
        split="train",
        tokenizer_name=model_name,
        max_length=256)  # max_length is only capped to speed-up example.
    val_dataset = ProtBertDataset(data_path=data_root,
                                  split="valid",
                                  tokenizer_name=model_name,
                                  max_length=256)
    test_dataset = ProtBertDataset(data_path=data_root,
                                   split="test",
                                   tokenizer_name=model_name,
                                   max_length=256)

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=1,  # total number of training epochs
        per_device_train_batch_size=1,  # batch size per device during training
        per_device_eval_batch_size=10,  # batch size for evaluation
        warmup_steps=1000,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=200,  # How often to print logs
        do_train=True,  # Perform training
        do_eval=True,  # Perform evaluation
        evaluation_strategy="epoch",  # evalute after eachh epoch
        gradient_accumulation_steps=
        64,  # total number of steps before back propagation
        fp16=True,  # Use mixed precision
        fp16_opt_level="02",  # mixed precision mode
        run_name="ProBert-BFD-MS",  # experiment name
        seed=3  # Seed for experiment reproducibility 3x3
    )

    trainer = Trainer(
        model_init=
        model_init,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # evaluation metrics
    )

    trainer.train()
    trainer.save_model('models/')
    predictions, label_ids, metrics = trainer.predict(test_dataset)