import torch.nn as nn
from transformers import BertModel


class ProteinClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ProteinClassifier, self).__init__()
        PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert_bfd_localization'
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.bert.config.hidden_size, n_classes))
        self.loss1 = nn.MultiLabelMarginLoss()
        self.loss1 = nn.MultiLabelSoftMarginLoss()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(output.pooler_output)
