import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
from IPython.display import display

class Model():

  def __init__(self):
    self.model = None
    self.device = None
    self.eval_type = None
    self.task_type = None
    self.model_name = None

  def reset_model(self):
    """
      Resets the model to its initial state
    """
    raise NotImplementedError("This method must be implemented in the derived class")

  def fit(self, data):
    """
      Virtual fit method
    """
    raise NotImplementedError("This method must be implemented in the derived class")

  def pipeline(self, data, nb_epochs, save_model=False):
    """
      All the train and test pipeline is implemented here.
      nb_epochs is the number of epochs for which we stop the training
      Warning : This is not what we will use to choose the number of epoch !!!!
      This method trains the model for nb_epochs, then saves the weights, and test the model / saves the scores
      Random & Out-of-event : Basic training and testing
    """
    if self.eval_type in ["random", "out-of-event"]:
      if self.eval_type == "random":
        print(f"Pipeline for {self.model_name} with a random splitting")
      else:
        print(f"Pipeline for {self.model_name} with an out-of-event framework")
      print("")
      self.fit(data.train_data, nb_epochs=nb_epochs)
      if save_model:
        self.save_model(epoch=nb_epochs)
      print("")
      print("Model Testing...")
      print("")
      acc, f1 = self.test(data.test_data)
      scores = self.model_scores(acc, f1)
      self.save_scores(scores, epoch=nb_epochs)
      print("")
      print("Scores")
      print("")
      display(scores)
      print("")
      print("")
    elif self.eval_type == "out-of-type":
      print(f"Pipeline for {self.model_name} with an out-of-type framework")
      print("")
      scores = []
      for crisis in data.data.keys():
        print(f"Isolated crisis : {crisis}")
        self.reset_model()
        # Séparer le dataset en 1 dataset de train et 1 dataset de test
        test_data = {crisis: data.data[crisis]}
        train_data = {task: dataset for task, dataset in data.data.items() if task != crisis}
        # Entraîner le modèle
        self.fit(train_data, nb_epochs=nb_epochs)
        if save_model:
          self.save_model(epoch=nb_epochs, isolated_crisis=crisis)
        # Tester le modèle
        print("")
        print("Model Testing...")
        acc, f1 = self.test(test_data, with_mean=False)
        scores.append(self.model_scores(acc, f1))

      # Concaténer les scores
      scores = pd.concat(scores, axis=0)
      scores.loc['Average'] = scores.mean()
      self.save_scores(scores, epoch=nb_epochs)
      print("")
      print("Scores")
      print("")
      display(scores)
      print("")
      print("")

  def build_path(self, to_save="model", epoch=None, isolated_crisis=""):
    """
      Builds the path for saving the model and the scores
    """
    if to_save == "model":
      return f"/content/drive/My Drive/CNRS@CREATE/Datasets/FrenchCorpus/Models/{self.task_type}_{self.eval_type}{isolated_crisis}_{self.model_name}_e{epoch}.pt"
    elif to_save == "scores":
      return f"/content/drive/My Drive/CNRS@CREATE/Datasets/FrenchCorpus/Results/{self.task_type}_{self.eval_type}_{self.model_name}_e{epoch}.csv"
    else:
      raise ValueError("Invalid save type")

  def hyper_parameter_tuning(self, train_data, test_data, nb_epochs, incremental=False):
    """
      This is what is used to choose the number of epochs for the model
    """
    if not incremental:
      for e in range(nb_epochs):
        print(f"{e+1} Epochs")
        print("")
        self.fit(train_data, nb_epochs=1)
        self.save_model(epoch=e+1)
        print("")
        print("Model Testing...")
        print("")
        acc, f1 = self.test(test_data)
        scores = self.model_scores(acc, f1)
        self.save_scores(scores, epoch=e+1)
        print("")
        print("Scores")
        print("")
        display(scores)
        print("")
        print("")
    else:
      for e in range(nb_epochs):
        self.reset_model()
        print(f"{e+1} Epochs")
        print("")
        self.fit(train_data, nb_epochs=e+1)
        self.save_model(epoch=e+1)
        print("")
        print("Model Testing...")
        print("")
        acc, f1 = self.test(test_data)
        scores = self.model_scores(acc, f1)
        self.save_scores(scores, epoch=e+1)
        print("")
        print("Scores")
        print("")
        display(scores)
        print("")
        print("")

  def test(self, data, with_mean=True):
    """
      Test function :
        - Returns the accuracies and f1_scores for each test crisis
    """

    self.model.eval()
    accuracies = {}
    f1_scores = {}

    for task in data.keys():

      progress_bar = tqdm(data[task], desc=f"Task : {task}")
      all_preds = None
      all_labels = None

      for batch in progress_bar:

        x, y = batch
        x, y = [input.to(self.device) for input in x], [target.to(self.device) for target in y]
        outputs = self.model(*x)
        preds = [output.argmax(dim=1) for output in outputs]

        if all_preds is None:
          all_preds = [[] for _ in preds]
          all_labels = [[] for _ in y]

        for i in range(len(preds)):
          all_preds[i].append(preds[i].detach().cpu())
          all_labels[i].append(y[i].detach().cpu())

      # Concat all predictions and labels
      accs = {}
      f1s = {}
      for i in range(len(all_preds)):
          y_pred = torch.cat(all_preds[i], dim=0)
          y_true = torch.cat(all_labels[i], dim=0)

          name = self.task_type[i]
          accs[name] = accuracy_score(y_true.numpy(), y_pred.numpy())
          f1s[name] = f1_score(y_true.numpy(), y_pred.numpy(), average="macro")

      accuracies[task] = accs
      f1_scores[task] = f1s

    if with_mean:
        acc_mean, f1_mean = {}, {}
        crises = list(accuracies.keys())             # sans "Average" pour l’instant
        for name in self.task_type:
            acc_mean[name] = np.mean([accuracies[c][name] for c in crises])
            f1_mean[name] = np.mean([f1_scores[c][name] for c in crises])

        accuracies["Average"] = acc_mean
        f1_scores["Average"]  = f1_mean

    return accuracies, f1_scores

  def model_scores(self, accuracies, f1_scores):
    """
      Creates a DataFrame to store the scores of the model
    """
    # On combine les deux dictionnaires
    combined = {}
    for crisis in accuracies.keys():
        combined[crisis] = {}
        for task_name, acc in accuracies[crisis].items():
            combined[crisis][("accuracy", task_name)] = acc
        for task_name, f1 in f1_scores[crisis].items():
            combined[crisis][("f1", task_name)] = f1

    # Création du DataFrame
    df = pd.DataFrame.from_dict(combined, orient="index")

    # Ordre propre des colonnes
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["metric", "task"])
    df = df.sort_index(axis=1)  # trie les colonnes par métrique puis tâche

    return df

  def save_scores(self, scores, epoch):
    """
      Saves the model scores in the path specified
    """
    scores.to_csv(self.build_path("scores", epoch))

  def save_model(self, epoch, isolated_crisis=""):
    """
      Saves the model in the path specified
    """
    torch.save(self.model.state_dict(), self.build_path("model", epoch, isolated_crisis))

  def load_model(self, epoch, isolated_crisis=""):
    """
      Loads the model from the path specified
    """
    self.model.load_state_dict(torch.load(self.build_path("model", epoch, isolated_crisis)))