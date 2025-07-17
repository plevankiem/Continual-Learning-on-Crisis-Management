import torch.nn as nn
import os

from transformers import RobertaModel, AutoModel

path_dir = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.join(path_dir, 'weights')

class FlauBERTClassifier(nn.Module):
    def __init__(self, nb_classes, device):
        super(FlauBERTClassifier, self).__init__()

        self.nb_classes = nb_classes
        self.pt_model = AutoModel.from_pretrained(os.path.join(path_dir, 'flaubert_fine_tuned_alldata'))
        self.feat_dim = self.pt_model.config.hidden_size
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

class RoBERTaClassifier(nn.Module):
    def __init__(self, nb_classes, device):
        super(RoBERTaClassifier, self).__init__()

        self.nb_classes = nb_classes
        self.model_name = "roberta-base"
        self.pt_model = RobertaModel.from_pretrained('roberta-base')
        self.feat_dim = self.pt_model.config.hidden_size
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