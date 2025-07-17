import os
import sys

from data.scripts.preprocessing import DataPreprocessing

from torch.utils.data import DataLoader, Dataset

class MultiTaskDataset(Dataset):
  """
    Dataset class for a Multi Task NLP Dataset.
    Warning : it represents only 1 dataset (1 task)
    Therefore, the full dataset will be a dictionary whose keys are the different tasks and values are TRECISDataset objects.
  """

  def __init__(self, inputs, targets):
    """
    inputs : should be a list of torch.tensor that represents all the desirable input for the model
    targets : should be a list of torch.tensor
    target_type : should be the key of targets corresponding to what type of labels we want to use : "category" or "priority" or "both"
    """
    self.inputs = inputs
    self.targets = targets

  def __len__(self):
    return len(self.targets[0])

  def __getitem__(self, idx):
    x = [self.inputs[i][idx] for i in range(len(self.inputs))]
    y = [self.targets[i][idx] for i in range(len(self.targets))]
    return (x, y)

class CrisisDataset():

  def __init__(self, tokenizer, batch_size, task_type=["humanitarian", "urgency", "utility"], dataset="HumAid"):

    self.batch_size = batch_size
    self.task_type = task_type
    self.dataset = dataset
    self.tokenizer = tokenizer

    data_preprocessing = DataPreprocessing(tokenizer, task_type, dataset)
    self.nb_classes = data_preprocessing.nb_classes
    self.data = {}
    for crisis, events in data_preprocessing.data.items():
        self.data[crisis] = {}
        for event, dataset in events.items():
            self.data[crisis][event] = MultiTaskDataset(dataset["inputs"], dataset["targets"])
            self.data[crisis][event] = DataLoader(self.data[crisis][event], shuffle=True, batch_size=batch_size)

    print("")
    print(f"✅ Dataset created with {len(self.data)} crisis")