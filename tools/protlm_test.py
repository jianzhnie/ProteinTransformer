import sys

from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import Trainer

from deepfold.data.protein_dataset import ProtBertDataset
from deepfold.models.transformers.multilabel_transformer import \
    BertForMultiLabelSequenceClassification

sys.path.append('../')


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    return {'auc': auc, 'ap': ap}


if __name__ == '__main__':
    model_name = 'Rostlab/prot_bert_bfd'
    data_root = '/home/niejianzheng/xbiome/datasets/protein'
    train_dataset = ProtBertDataset(
        data_path=data_root,
        split='train',
        tokenizer_name=model_name,
        max_length=512)  # max_length is only capped to speed-up example.
    val_dataset = ProtBertDataset(data_path=data_root,
                                  split='valid',
                                  tokenizer_name=model_name,
                                  max_length=512)
    test_dataset = ProtBertDataset(data_path=data_root,
                                   split='test',
                                   tokenizer_name=model_name,
                                   max_length=512)
    num_classes = train_dataset.num_classes
    # Load trained model
    model_path = 'work_dir/checkpoint-'
    model = BertForMultiLabelSequenceClassification.from_pretrained(
        model_path, num_labels=num_classes)
    # Define test trainer
    test_trainer = Trainer(model)
    predictions, label_ids, metrics = test_trainer.predict(val_dataset)
