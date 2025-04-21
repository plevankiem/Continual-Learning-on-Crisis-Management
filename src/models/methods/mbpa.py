from generic_class import Model
from pretrained_models import FlauBERTClassifier, FlauBERTEncoder

from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from itertools import islice

class ReplayBuffer:
    def __init__(self, capacity=None):
      self.capacity = capacity
      if capacity:
        self.buffer = deque(maxlen=capacity)
      else:
        self.buffer = deque()

    def add(self, input_ids, attention_mask, y):
      """Add a new sample to the buffer."""
      for i in range(input_ids.size(0)):
        self.buffer.append((input_ids[i:i+1].detach().cpu(), attention_mask[i:i+1].detach().cpu(), y[i:i+1].detach().cpu()))

    def sample(self, batch_size):
      """Randomly sample a batch of (x, y) tensors."""
      batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
      input_ids, attention_mask, ys = zip(*batch)
      return torch.cat(input_ids), torch.cat(attention_mask), torch.cat(ys)

    def find_neighbors(self, u, K, encoder, device, batch_size):
      """
      Find K nearest neighbors in memory for each vector in u.

      Args:
          u (Tensor): (B, D) query vectors from encoder
          K (int): number of neighbors to retrieve
          encoder (nn.Module): encoder model
          device (torch.device): target device (usually GPU)
          batch_size (int): memory encoding batch size

      Returns:
          List[List[Tuple]]: K neighbors for each query in u
      """
      buffer_list = list(self.buffer)  # Convert deque to list for slicing
      encoded_xs = []

      # Encode the memory buffer in batches
      for i in range(0, len(buffer_list), batch_size):
          batch = buffer_list[i:i + batch_size]
          input_ids_batch = [item[0] for item in batch]
          attention_mask_batch = [item[1] for item in batch]

          input_ids_batch = torch.cat(input_ids_batch, dim=0).to(device)
          attention_mask_batch = torch.cat(attention_mask_batch, dim=0).to(device)

          with torch.no_grad():
              encoded = encoder(input_ids_batch, attention_mask_batch)
              encoded_xs.append(encoded)

          # Free up memory
          del input_ids_batch, attention_mask_batch, encoded
          torch.cuda.empty_cache()

      # Combine all encoded memory items
      encoded_xs = torch.cat(encoded_xs, dim=0)  # (N, D)

      # Make sure queries are on the same device
      u = u.to(device)  # (B, D)

      # Compute pairwise distances: (B, N)
      dists = torch.norm(encoded_xs - u, dim=1)  # (N,)

      # Get K nearest
      nearest_indices = torch.topk(dists, k=K, largest=False).indices
      neighbors = [buffer_list[i] for i in nearest_indices.tolist()]

      del encoded_xs
      torch.cuda.empty_cache()
      gc.collect()

      return neighbors

    def __len__(self):
      return len(self.buffer)

class MbPA(Model):

  def __init__(self, R, lambda_, batch_size, K, L, device, nb_classes, store_frequency):
    super(MbPA, self).__init__()

    self.R = R
    self.lambda_ = lambda_ # Regularisation term
    self.batch_size = batch_size
    self.L = L # Inference steps
    self.K = K
    self.device = device
    self.model = FlauBERTClassifier(nb_classes=nb_classes).to(device)
    self.encoder = FlauBERTEncoder().to(device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)
    self.criterion = nn.CrossEntropyLoss(reduction='sum')
    self.memory = ReplayBuffer(capacity=None)
    self.store_frequency = store_frequency

  def inference_loss(self, output, target, old_params):
    """
    Compute the inference loss
    """
    ce_loss = F.cross_entropy(output, target, reduction="mean")

    reg_loss = 0.0
    for p_new, p_old in zip(self.model.parameters(), old_params):
        reg_loss += torch.norm(p_new - p_old, p=2) ** 2

    total_loss = ce_loss + self.lambda_ * reg_loss
    return total_loss

  def fit(self, data, nb_epochs):
    """
    Input :
      - data : sequence of DataLoader which contain the data from each dataset
    Output :
      - None (Changes the model's weights)
    """

    losses = []
    memory_losses = []
    for task in data.keys():

      for epoch in range(nb_epochs):

        task_loss = 0.0
        memory_loss = 0.0
        loss_s = None
        progress_bar = tqdm(data[task], desc=f"Task : {task} | Epoch {epoch+1}")
        for i, batch in enumerate(progress_bar):
          self.model.train()

          # Samples from memory
          if (i+1) % self.R == 0:
            ids_s, am_s, y_s = self.memory.sample(self.batch_size)
            self.optimizer.zero_grad()
            ids_s, am_s, y_s = ids_s.to(self.device), am_s.to(self.device), y_s.to(self.device)
            outputs_s = self.model(ids_s, am_s)
            loss_s = self.criterion(outputs_s, y_s)
            memory_loss += loss_s.item()
            loss_s.backward()
            self.optimizer.step()
            del ids_s, am_s, y_s, outputs_s

          # Regular training step
          ids, am, y = batch
          self.optimizer.zero_grad()
          ids, am, y = ids.to(self.device), am.to(self.device), y.to(self.device)
          outputs = self.model(ids, am)
          loss = self.criterion(outputs, y)
          task_loss += loss.item()
          loss.backward()
          self.optimizer.step()

          # Store samples in memory
          if i % self.store_frequency == 0 and epoch == 0:
            self.memory.add(ids, am, y)

          if loss_s:
            progress_bar.set_postfix({"Task Loss": task_loss / (i+1), "Memory Loss": loss_s.item()})
          else:
            progress_bar.set_postfix({"Task Loss": task_loss / (i+1)})

          del ids, am, y, outputs, loss

          torch.cuda.empty_cache()
          gc.collect()

        losses.append(task_loss)
        memory_losses.append(memory_loss)


    return losses, memory_losses

  def test(self, data):
    """
    Input :
      - data : same as in fit function
    Output :
      - prediction : sequence of predictions for each dataset (list of tuples that contain tensors)
    """
    output = {}
    for task in data.keys():
      predictions = []
      y_true = []
      dataloader = DataLoader(data[task].dataset, shuffle=True, batch_size=1)
      progress_bar = tqdm(islice(dataloader, 100), total=100, desc=f"Task : {task}")

      for ids, am, y in progress_bar:
        old_model_parameters = [p.clone().detach() for p in self.model.parameters()]
        ids, am, y = ids.to(self.device), am.to(self.device), y.to(self.device)
        adaptation_optimizer = torch.optim.SGD(self.model.parameters(), lr=5e-3)

        # Find the nearest neighbors in the memory
        with torch.no_grad():
          u = self.encoder(ids, am)
        neighbors = self.memory.find_neighbors(u, self.K, self.encoder, self.device, batch_size=64)
        ids_n, am_n, y_n = torch.cat([neighbor[0] for neighbor in neighbors]), torch.cat([neighbor[1] for neighbor in neighbors]), torch.cat([neighbor[2] for neighbor in neighbors])
        ids_n, am_n, y_n = ids_n.to(self.device), am_n.to(self.device), y_n.to(self.device)

        # Local Adaptation
        for l in range(self.L):
          adaptation_optimizer.zero_grad()
          outputs_n = self.model(ids_n, am_n)
          loss = self.inference_loss(outputs_n, y_n, old_model_parameters)
          loss.backward()
          adaptation_optimizer.step()

        # Make the prediction
        with torch.no_grad():
          y_pred = self.model(ids, am).argmax(dim=-1)

        predictions.append(y_pred.item())
        y_true.append(y.item())

        # Set parameters back to normal
        for p, old_p in zip(self.model.parameters(), old_model_parameters):
          p.data.copy_(old_p)

        # Delete local variables after use
        del adaptation_optimizer
        del ids, am, y, ids_n, am_n, y_n, outputs_n, loss
        torch.cuda.empty_cache()

      output[task] = (torch.tensor(predictions).cpu().numpy(), torch.tensor(y_true).cpu().numpy())

      torch.cuda.empty_cache()  # clears unused memory
      gc.collect()

    accuracies, f1_scores = {}, {}
    for task in data.keys():
      accuracies[task] = accuracy_score(output[task][1], output[task][0])
      f1_scores[task] = f1_score(output[task][1], output[task][0], average="macro")
    y_true_concat = np.concatenate([output[task][1] for task in data.keys()])
    y_pred_concat = np.concatenate([output[task][0] for task in data.keys()])
    accuracies["All tasks"] = accuracy_score(y_true_concat, y_pred_concat)
    f1_scores["All tasks"] = f1_score(y_true_concat, y_pred_concat, average="macro")

    return accuracies, f1_scores