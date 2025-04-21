import torch
import torch.nn as nn
import plotly.graph_objects as go
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
from utils import TextPreprocessing, BertInput

class DataPreprocessing():

  def __init__(self, path, tokenizer, task=["utility"], eval_type="random", event_test_set=0):
    """
      paths : list of paths to the json files
      Initializes the dataset dictionary and the JSON objects.
    """
    self.task = task
    self.eval_type = eval_type
    self.events_set_test = [
        ["Aude", "AttaqueTrebes", "Beryl-Guadeloupe", "EffondrementLille", "ExplosionSanary", "Incendie"],
        ["Corse", "Corse-fionn", "EffondrementMarseille", "IncendieLubrizol", "NotreDame"],
        ["Autre", "IncendieLubrizol", "Irma", "NotreDame", "EffondrementLille"],
    ][event_test_set]
    self.text_preprocessor = TextPreprocessing()
    self.bert_input = BertInput(tokenizer, max_length=146)
    self.classes = {
        "humanitarian": ("CAT", {'Message-NonUtilisable': 0, 'Soutiens': 1, 'AutresMessages': 2, 'Critiques': 3, 'Avertissement-conseil': 4, 'Degats-Materiels': 5, 'Degats-Humains': 6, 'NotAnnotated': 7}),
        "urgency": ("CAT3", {'Message-NonUtilisable': 0, 'Message-InfoNonUrgent': 1, 'Message-InfoUrgent': 2}),
        "utility": ("CAT2", {'Message-NonUtilisable': 0, 'Message-Utilisable': 1})
    }
    self.nb_classes = [len(self.classes[type_class][1]) for type_class in self.task]
    self.df = pd.read_csv(path, sep="\t", low_memory=False)
    self.clean_dataframe()
    self.create_dictionary()
    self.create_datasets()
  
  def clean_dataframe(self):
    """
      Clean the Data Frame containing all the data
      Remove the not annotated tweets
    """
    self.df = self.df[self.df["CAT"] != "NotAnnotated"]

  def create_dictionary(self):
    """
      Creates the giant dictionary with {
        crisis_type : {
          event : {
            inputs: [torch.tensor],
            targets: [torch.tensor]
          }
        }
      }
    """
    self.dictionary = {}
    for crisis in tqdm(self.df["type_crisis"].unique()):
      # print(f"Processing crisis type : {crisis}")
      self.dictionary[crisis] = {}
      for event in self.df[self.df["type_crisis"] == crisis]["event"].unique():
        # print(f"    {event}")
        df_event = self.df[self.df["event"] == event]
        self.dictionary[crisis][event] = {
            "inputs": self.bert_input.fit_transform(list(self.text_preprocessor.preprocess_text(df_event["processed_text"]))),
            "targets": [ torch.tensor(list(df_event[self.classes[type_class][0]].map(self.classes[type_class][1]))) for type_class in self.task ]
        }
        # print(f"        Nb targets : {len(self.dictionary[crisis][event]['targets'])}")
        # print(f"        Targets values : {self.dictionary[crisis][event]['targets'][0].unique()}")
        # print(f"        Len inputs : {len(self.dictionary[crisis][event]['inputs'])}")
        # print(f"        Len targets : {len(self.dictionary[crisis][event]['targets'][0])}")

  def concatenate_events(self, events):
    """
      Concatenates the data from the two events
    """
    events_concatenated = {}
    if len(events) == 0:
      return None
    elif len(events) == 1:
      return events[0]
    else:
      for key in events[0].keys():
        events_concatenated[key] = []
        # inputs or targets
        for variable in range(len(events[0][key])):
          # each input/target
          events_concatenated[key].append(torch.cat([event[key][variable] for event in events], dim=0))
      return events_concatenated

  def create_datasets(self):
    """
      Random + Out-of-event :
        This function creates self.train_data and self.test_data that are dictionary containing for each crisis a { 'inputs': [], 'targets : [] } dictionary.
        All the torch.tensors have been concatenated. The separation between train and test is done accordingly to the eval method.
      Out-of-type :
        This function creates self.data that is a dictionary containing for each crisis a { 'inputs': [], 'targets : [] } dictionary.
        There is no separation between train and test set yet.
    """
    if self.eval_type == "random":
      self.train_data = {}
      self.test_data = {}
      for crisis, events in self.dictionary.items():
        # Contatenate all events
        concatenated_events = self.concatenate_events(list(events.values()))
        # Regroup all the tensors to split
        all_data = concatenated_events["inputs"] + concatenated_events["targets"]
        # Split
        train_split, test_split = train_test_split(
            list(zip(*all_data)),
            test_size=0.2,
            random_state=42
        )
        # Dézipper the lists
        train_split = list(zip(*train_split))
        test_split = list(zip(*test_split))

        # Rebuild the dictionaries
        self.train_data[crisis] = {
            "inputs": [torch.stack(feature) for feature in train_split[:len(concatenated_events["inputs"])]],
            "targets": [torch.stack(label) for label in train_split[len(concatenated_events["inputs"]):]]
        }
        self.test_data[crisis] = {
            "inputs": [torch.stack(feature) for feature in test_split[:len(concatenated_events["inputs"])]],
            "targets": [torch.stack(label) for label in test_split[len(concatenated_events["inputs"]):]]
        }

    elif self.eval_type == "out-of-event":
      self.train_data = {}
      self.test_data = {}
      for crisis, events in self.dictionary.items():
        # Contatenate train events
        train_events_list = [event_data for event_key, event_data in events.items() if event_key not in self.events_set_test]
        train_concatenated_events = self.concatenate_events(train_events_list)
        if len(train_events_list) > 0:
          self.train_data[crisis] = train_concatenated_events
        # Choose test event
        test_events_list = [event_data for event_key, event_data in events.items() if event_key in self.events_set_test]
        test_events_names = [event_key for event_key, event_data in events.items() if event_key in self.events_set_test]
        test_concatenated_events = self.concatenate_events(test_events_list)
        if len(test_events_names) == 1:
          self.test_data[f"{crisis} | {test_events_names[0]}"] = test_concatenated_events
        elif len(test_events_names) > 1:
          self.test_data[f"{crisis} | {test_events_names}"] = test_concatenated_events

    elif self.eval_type == "out-of-type":
      self.data = {}
      for crisis, events in self.dictionary.items():
        concatenated_events = self.concatenate_events(list(events.values()))
        self.data[crisis] = concatenated_events

    else:
      raise ValueError("Invalid evaluation type")