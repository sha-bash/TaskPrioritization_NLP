import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig

class BERTPriorityClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", num_classes=2):
        super(BERTPriorityClassifier, self).__init__()
        # Загрузка предобученной модели BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # Прямой проход через BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Выход последнего CLS токена
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
