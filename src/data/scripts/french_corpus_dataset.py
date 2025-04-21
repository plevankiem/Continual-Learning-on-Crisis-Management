import torch
import torch.nn as nn
import plotly.graph_objects as go
from IPython.display import display
import numpy as np
import pandas as pd
import os
import glob
import gc
import json
from time import time
from transformers import BertTokenizer, BertTokenizerFast, BertModel, FlaubertModel, FlaubertTokenizer
from tqdm.auto import tqdm
from collections import deque, namedtuple
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from itertools import islice
import re
import sacremoses

from data_preprocessing import DataPreprocessing

class FrenchCorpusDataset(Dataset):
  """
    Dataset class for the TRECIS Dataset.
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

class FullFrenchCorpusDataset():

  def __init__(self, path, tokenizer, batch_size, task=["2-cat"], eval_type="random", event_test_set=0):

    self.batch_size = batch_size
    self.task_type = task
    self.eval_type = eval_type

    data_preprocessing = DataPreprocessing(path, tokenizer, task, eval_type, event_test_set)
    self.nb_classes = data_preprocessing.nb_classes
    self.tokenizer = data_preprocessing.text_preprocessor.tokenizer
    if eval_type == "out-of-type":
      self.data = {}
      for crisis, dataset in data_preprocessing.data.items():
        self.data[crisis] = FrenchCorpusDataset(dataset["inputs"], dataset["targets"])
        self.data[crisis] = DataLoader(self.data[crisis], shuffle=True, batch_size=batch_size)
      self.data = dict(sorted(self.data.items()))
      print("")
      print(f"✅ Dataset created with {len(self.data)} crisis")
    else:
      self.train_data = {}
      self.test_data = {}
      for task, dataset in data_preprocessing.train_data.items():
        self.train_data[task] = FrenchCorpusDataset(dataset["inputs"], dataset["targets"])
        self.train_data[task] = DataLoader(self.train_data[task], shuffle=True, batch_size=batch_size)
      for task, dataset in data_preprocessing.test_data.items():
        self.test_data[task] = FrenchCorpusDataset(dataset["inputs"], dataset["targets"])
        self.test_data[task] = DataLoader(self.test_data[task], shuffle=True, batch_size=batch_size)

      self.test_data = dict(sorted(self.test_data.items()))
      self.train_data = dict(sorted(self.train_data.items()))

      print("")
      print(f"✅ Dataset created with {len(self.train_data)} train crisis and {len(self.test_data)} test crisis")