import torch.nn as nn
from transformers import RobertaModel

class RoBERTaClassifier(nn.Module):
    def __init__(self, nb_classes, device):
        super(RoBERTaClassifier, self).__init__()

        self.nb_classes = nb_classes
        self.model_name = "roberta-base"
        self.pt_model = RobertaModel.from_pretrained('roberta-base')
        self.feat_dim = self.pt_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.fcs = nn.ModuleList([nn.Linear(self.feat_dim, class_dim).to(device) for class_dim in self.nb_classes])

    def forward(self, input_ids, attention_mask):
        outputs = self.pt_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = [self.fcs[i](pooled_output) for i in range(len(self.fcs))]
        return logits

    def encode(self, input_ids, attention_mask):
        outputs = self.pt_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return pooled_output

    def classify(self, pooled_output):
        logits = [self.fcs[i](pooled_output) for i in range(len(self.fcs))]
        return logits