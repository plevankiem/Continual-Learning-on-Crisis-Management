from generic_class import Model
from pretrained_models import FlauBERTClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import tqdm

class MergedModel(Model):

  def __init__(self, data, device):
    super(MergedModel, self).__init__()

    self.device = device
    self.nb_classes = data.nb_classes
    self.eval_type = data.eval_type
    self.task_type = data.task_type
    self.model_name = "merged"
    self.tokenizer = data.tokenizer
    self.model = FlauBERTClassifier(nb_classes=data.nb_classes, tokenizer=self.tokenizer, device=device).to(device)
    self.nb_params = sum([p.numel() for p in self.model.parameters()])
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)

  def reset_model(self):
    self.model = FlauBERTClassifier(nb_classes=self.nb_classes, tokenizer=self.tokenizer, device=self.device).to(self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)

  def criterion(self, outputs, targets):
    losses = [ F.cross_entropy(outputs[i], targets[i], reduction="sum") for i in range(len(outputs)) ]
    return sum(losses) / len(losses)

  def fit(self, data, nb_epochs):
    """
    Input :
      - data : sequence of DataLoader which contain the data from each dataset
    Output :
      - None (Changes the model's weights)
    """

    # Create the new dataset
    datasets = [loader.dataset for loader in data.values()]
    concat_dataset = torch.utils.data.ConcatDataset(datasets)
    dataset = torch.utils.data.DataLoader(concat_dataset, batch_size=64, shuffle=True)

    # Training part
    self.model.train()
    losses = []
    for epoch in range(nb_epochs):
      total_loss = 0.0
      progress_bar = tqdm(dataset, desc=f"Epoch {epoch+1}")

      for i, batch in enumerate(progress_bar):

        self.optimizer.zero_grad()
        x, y = batch
        x, y = [input.to(self.device) for input in x], [target.to(self.device) for target in y]
        outputs = self.model(*x)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": total_loss / (i+1)})

      losses.append(total_loss / len(dataset))
      del x, y, outputs, loss
      torch.cuda.empty_cache()
      gc.collect()