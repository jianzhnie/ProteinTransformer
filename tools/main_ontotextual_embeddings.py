import argparse
import sys

import numpy as np
import torch
from datasets import load_metric
from transformers import (AutoModelForSequenceClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from deepfold.data.ontotextual_dataset import OntoTextDataset

sys.path.append('../')

sys.path.append('../')

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


parser = argparse.ArgumentParser(
    description='Protein function Classification Model Train config')
parser.add_argument('--data_path',
                    default='',
                    type=str,
                    help='data dir of dataset')
parser.add_argument(
    '--tokenizer_dir',
    default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    type=str,
    help='tokenizer_dir')
parser.add_argument(
    '--pretrain_model_dir',
    default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    type=str,
    help='pretrained model checkpoint dir')
parser.add_argument('--epochs',
                    default=90,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256) per gpu')
parser.add_argument('--output-dir',
                    default='./work_dirs',
                    type=str,
                    help='output directory for model and log')


def main(args):
    if has_wandb:
        wandb.init(project='Bert-For-GoTerm-Embedding', entity='jianzhnie')
    # Dataset and DataLoader
    dataset = OntoTextDataset(data_dir=args.data_path,
                              tokenizer_name=args.tokenizer_dir,
                              max_length=512)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrain_model_dir, num_labels=3)

    training_args = TrainingArguments(
        report_to='wandb',
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.epochs,  # total number of training epochs
        per_device_train_batch_size=args.
        batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.
        batch_size,  # batch size for evaluation
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
        logging_dir=args.output_dir,  # directory for storing logs
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
    trainer.save_model(args.output_dir + '/' + 'models')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
