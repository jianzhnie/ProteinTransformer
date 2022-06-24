import argparse
import sys

import numpy as np
import torch
from datasets import load_metric
from transformers import (AutoModelForSequenceClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)
sys.path.append('../')
from deepfold.data.ontotextual_dataset import OntoTextDataset



try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

metric = load_metric('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    if has_wandb:
        wandb.init(project='Bert-For-GoTerm-Embedding', entity='jianzhnie')
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    data_root = '/home/niejianzheng/xbiome/datasets/protein'
    # Dataset and DataLoader
    dataset = OntoTextDataset(data_dir=data_root,
                              tokenizer_name=model_name,
                              max_length=512)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    # model
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=3)

    training_args = TrainingArguments(
        report_to='wandb',
        output_dir='./work_dir',  # output directory
        num_train_epochs=20,  # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=128,  # batch size for evaluation
        learning_rate=2e-5,  # learning_rate
        warmup_steps=1000,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        gradient_accumulation_steps=1,
        # total number of steps before back propagation
        fp16=True,  # Use mixed precision
        fp16_opt_level='02',  # mixed precision mode
        do_train=True,  # Perform training
        do_eval=True,  # Perform evaluation
        save_strategy='epoch',  # save model every epoch
        evaluation_strategy='epoch',  # evalute after each epoch
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        run_name='Bert-For-GoTerm-Embedding',  # experiment name
        logging_dir='./logs',  # directory for storing logs
        logging_steps=100,  # How often to print logs
        seed=3  # Seed for experiment reproducibility 3x3
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # evaluation metrics
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model('models/')


if __name__ == '__main__':
    main()
