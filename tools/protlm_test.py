import pickle
import sys

from transformers import Trainer

from deepfold.core.metrics.custom_metrics import compute_roc
from deepfold.data.protein_dataset import ProtBertDataset
from deepfold.models.transformers.multilabel_transformer import \
    BertForMultiLabelSequenceClassification

sys.path.append('../')


def compute_metrics(predictions):
    labels = predictions.label_ids
    preds = predictions.predictions
    roc = compute_roc(labels, preds)
    return {'roc': roc}


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
    model_path = 'work_dir/checkpoint-6900'
    model = BertForMultiLabelSequenceClassification.from_pretrained(
        model_path, num_labels=num_classes)
    # Define test trainer
    test_trainer = Trainer(model)
    predictions = test_trainer.predict(test_dataset)
    results = compute_metrics(predictions)
    print(results)
    results_path = 'work_dir/predictions.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(predictions, f)
