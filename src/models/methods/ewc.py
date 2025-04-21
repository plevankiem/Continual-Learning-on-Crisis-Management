from generic_class import Model
from pretrained_models import FlauBERTClassifier

import torch
import torch.nn.functional as F
import gc
import tqdm

class ElasticWeightsConsolidation(Model):
  """
  Compressed version of EWC as we don't store all the set of parameters and all the Fisher matrix. It is commonly used when working with a large model like BERT.
  """
  def __init__(self, data, device, lambda_):
    super(ElasticWeightsConsolidation, self).__init__()

    self.device = device
    self.nb_classes = data.nb_classes
    self.eval_type = data.eval_type
    self.task_type = data.task_type
    self.model_name = "ewc"
    self.tokenizer = data.tokenizer
    self.model = FlauBERTClassifier(nb_classes=self.nb_classes, tokenizer=self.tokenizer, device=self.device).to(device)
    self.nb_params = sum([p.numel() for p in self.model.parameters()])
    self.fisher_matrix = [torch.zeros_like(p) for p in self.model.parameters()]
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)
    self.lambda_ = lambda_

  def reset_model(self):
    self.model = FlauBERTClassifier(nb_classes=self.nb_classes, tokenizer=self.tokenizer, device=self.device).to(self.device)
    self.fisher_matrix = [torch.zeros_like(p) for p in self.model.parameters()]
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)

  def reg_loss(self, old_params, first):
    """
    Compute the regularization term
    """
    reg_loss = 0.0
    if not first:
      for n, (p_new, p_old) in enumerate(zip(self.model.parameters(), old_params)):
          reg_loss += torch.sum(self.fisher_matrix[n] * (p_new - p_old)**2)
    return reg_loss

  def simple_criterion(self, outputs, targets):
      losses = [ F.cross_entropy(outputs[i], targets[i], reduction="sum") for i in range(len(outputs)) ]
      return sum(losses) / len(losses)

  def compute_fischer(self, dataset):
    """
    Compute the Fischer Matrix F for one dataset
    """
    self.model.eval()
    none_grad = 0

    fisher_matrix = [torch.zeros_like(p) for p in self.model.parameters()]

    progress_bar = tqdm(dataset, desc="    Computing the Fisher Matrix")
    for batch in progress_bar:

      # Reset the gradients
      self.model.zero_grad()

      # Compute the gradients
      x, y = batch
      x, y = [input.to(self.device) for input in x], [target.to(self.device) for target in y]
      outputs = self.model(*x)
      loss = self.simple_criterion(outputs, y)
      loss.backward()

      # Update the matrix
      batch_size = x[0].size(0)
      for n, p in enumerate(self.model.parameters()):
        if p.grad is not None:
          fisher_matrix[n] += batch_size * (p.grad.detach() ** 2) / len(dataset)
        else:
          none_grad += 1

      # Saving GPU memory
      del x, y, outputs, loss
      torch.cuda.empty_cache()
      gc.collect()

    return fisher_matrix

  def update_fisher(self, dataset):
    """Update the fisher matrix with the new dataset"""
    new_fisher_matrix = self.compute_fischer(dataset)
    for n in range(len(self.fisher_matrix)):
      self.fisher_matrix[n] += new_fisher_matrix[n]

  def fit(self, data, nb_epochs):
    """
    Input :
      - data : sequence of DataLoader which contain the data from each dataset
    Output :
      - None (Changes the model's weights)
    """
    self.model.train()
    losses = []
    for t, task in enumerate(data.keys()):
      loss = None
      old_params = [p.detach().clone() for p in self.model.parameters()]
      first = (t == 0)
      for epoch in range(nb_epochs):

        task_loss = 0.0
        progress_bar = tqdm(data[task], desc=f"Task : {task} | Epoch {epoch+1}")

        for i, batch in enumerate(progress_bar):

          self.optimizer.zero_grad()
          x, y = batch
          x, y = [input.to(self.device) for input in x], [target.to(self.device) for target in y]
          outputs = self.model(*x)
          simple_loss = self.simple_criterion(outputs, y)
          reg_loss = self.reg_loss(old_params, first)
          loss = simple_loss + self.lambda_ * reg_loss
          loss.backward()
          # loss = self.criterion(outputs, y, old_params)
          self.optimizer.step()
          task_loss += simple_loss.item()
          progress_bar.set_postfix({"Task Loss": task_loss / (i+1), "Fisher Loss": reg_loss.item() if not first else 0})

        losses.append(task_loss)
        del x, y, outputs, loss
        torch.cuda.empty_cache()
        gc.collect()

      self.update_fisher(data[task])

    return losses