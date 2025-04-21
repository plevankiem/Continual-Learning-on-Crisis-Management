import torch.nn as nn
from transformers import FlaubertModel

class FlauBERTClassifier(nn.Module):
    def __init__(self, nb_classes, tokenizer, device):
        super(FlauBERTClassifier, self).__init__()

        self.nb_classes = nb_classes
        self.pt_model = FlaubertModel.from_pretrained('flaubert/flaubert_base_uncased')
        self.pt_model.resize_token_embeddings(len(tokenizer))
        self.fcs = [nn.Linear(self.pt_model.config.hidden_size, class_dim).to(device) for class_dim in self.nb_classes]

    def forward(self, input_ids, attention_mask):
        outputs = self.pt_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = [fc(pooled_output) for fc in self.fcs]
        return logits

class FlauBERTEncoder(nn.Module):
    def __init__(self, tokenizer):
        super(FlauBERTEncoder, self).__init__()

        self.pt_model = FlaubertModel.from_pretrained('flaubert/flaubert_base_uncased')
        self.pt_model.resize_token_embeddings(len(tokenizer))
        self.pt_model.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        outputs = self.pt_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]