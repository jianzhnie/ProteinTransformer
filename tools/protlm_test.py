import sys
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import Trainer
import pickle
sys.path.append('../')
from deepfold.loss.custom_metrics import compute_mcc, compute_roc
from deepfold.data.protein_dataset import ProtBertDataset
from deepfold.models.transformers.multilabel_transformer import \
    BertForMultiLabelSequenceClassification



def compute_metrics(labels, preds):
    mcc = compute_mcc(labels, preds)
    roc = compute_roc(labels, preds)
    return {'mcc': mcc, 'roc': roc}


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
    labels = predictions.label_ids
    preds = predictions.predictions
    preds = preds > 0.5
    results = compute_metrics(labels, preds)
    print(results)
    results_path = 'work_dir/predictions.pkl'
    with open(results_path, "wb") as f:
        pickle.dump(predictions, f)