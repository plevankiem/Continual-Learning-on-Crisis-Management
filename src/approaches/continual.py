from src.utils import display_scores, stratified_dataloader_reduction, concatenate_events
from torch.utils.data import DataLoader, Subset
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import json
import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np


path = Path(os.path.dirname(__file__))
config_path = os.path.join(path.parent.parent, "config")

class ContinualLearning():

    def __init__(self, data):
        self.model = None
        self.device = None
        self.model_name = None
        self.dataset = data.dataset
        self.seeds = [42]
        self.protocol = "similarity"
        with open(os.path.join(config_path, "splits.json"), "r") as f:
            self.splits = json.load(f)

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

    def compute_cl_scores(self, test_dfs):
        """
        Computes all the CL Scores for the test set
        """
        crisis_list = [crisis for crisis in test_dfs[0].index if crisis != "Average"]
        # Calcul de la matrice des a_i_j pour chaque tâche
        scores_list = ["AIA"] + ["BWT"] + ["BWT" + f"_{crisis}" for crisis in test_dfs[0].index if crisis != "Average"] + ["FM"] + ["FM" + f"_{crisis}" for crisis in test_dfs[0].index if crisis != "Average"]
        cl_scores = pd.DataFrame(index=scores_list)

        A = { task: np.zeros((len(test_dfs), len(test_dfs))) for task in self.task_type }
        for i in range(len(test_dfs)):
            for j in range(len(test_dfs)):
                for task in self.task_type:
                    A[task][i, j] = test_dfs[i][[task]].iloc[j, 0]

        # Calcul des scores pour chaque tâche
        for task in self.task_type:
            cl_dico = {}
            # AIA
            aia = 0.0
            for k in range(len(test_dfs)):
                aia_spec = 0.0
                for j in range(k+1):
                    aia_spec += A[task][k, j]
                aia_spec /= (k+1)
                aia += aia_spec

            aia /= len(test_dfs)
            cl_dico["AIA"] = aia

            # FM
            fm = 0
            for j in range(len(test_dfs)-1):
                fm += max([ A[task][i, j] - A[task][-1, j] for i in range(len(test_dfs)-1)])
                cl_dico[f"FM_{crisis_list[j]}"] = max([ A[task][i, j] - A[task][-1, j] for i in range(len(test_dfs)-1)])
            fm /= (len(test_dfs) - 1)
            cl_dico["FM"] = fm

            # BWT
            bwt = 0
            for j in range(len(test_dfs)-1):
                bwt += A[task][j, j] - A[task][-1, j]
                cl_dico[f"BWT_{crisis_list[j]}"] = A[task][j, j] - A[task][-1, j]
            bwt /= (len(test_dfs) - 1)
            cl_dico["BWT"] = bwt

            cl_scores[task] = pd.Series(cl_dico)

        return cl_scores, A
    
    def split_dataset(self, data, n_set):
        train_data, test_data = {}, {}
        split_key = self.splits[self.dataset][n_set][0]
        split = self.splits[self.dataset][n_set][1]

        for crisis in data.data.keys():
            for event in data.data[crisis].keys():
                #if event == "hurricane_florence_2018":
                #event_dataloader = stratified_dataloader_reduction(data.data[crisis][event], total_samples=2000)
                if True:
                    event_dataloader = stratified_dataloader_reduction(data.data[crisis][event], total_samples=5)
                else:
                    event_dataloader = data.data[crisis][event]
                if split_key == "train":
                    if event in split:
                        if crisis in train_data:
                            train_data[crisis].append(event_dataloader)
                        else:
                            train_data[crisis] = [event_dataloader]
                    else:
                        if crisis in test_data:
                            test_data[crisis].append(event_dataloader)
                        else:
                            test_data[crisis] = [event_dataloader]
                
                elif split_key == "test":
                    if event in split:
                        if crisis in test_data:
                            test_data[crisis].append(event_dataloader)
                        else:
                            test_data[crisis] = [event_dataloader]
                    else:
                        if crisis in train_data:
                            train_data[crisis].append(event_dataloader)
                        else:
                            train_data[crisis] = [event_dataloader]
                else:
                    raise ValueError(f"Unknown split key: {split_key}")

        train_data = {crisis: concatenate_events(train_data[crisis]) for crisis in train_data.keys()}
        test_data = {crisis: concatenate_events(test_data[crisis]) for crisis in test_data.keys()}

        return train_data, test_data

    def pipeline(self, data, nb_epochs, idx=None):
        """
            All the train and test pipeline is implemented here.
            nb_epochs is the number of epochs for which we stop the training
            Warning : This is not what we will use to choose the number of epoch !!!!
            This method trains the model for nb_epochs, then saves the weights, and test the model / saves the scores
            Random & Out-of-event : Basic training and testing
        """
        print(f"Pipeline for {self.model_name} with a continual learning framework")
        print("")

        final_scores = []
        crisis_list = list(data.data.keys())
        save_dir = os.path.join(path.parent.parent, "results", self.dataset, self.model_name)

        if self.protocol == "similarity":
            all_orders = [
                ['wildfires', 'hurricane', 'floods', 'earthquake'],
                ['hurricane', 'earthquake', 'floods', 'wildfires'],
                ['earthquake', 'hurricane', 'floods', 'wildfires'],
                ['floods', 'hurricane', 'earthquake', 'wildfires']
            ]
        else:
            all_orders = [crisis_list]
            for _ in range(len(crisis_list)-1):
                all_orders.append(all_orders[-1][1:] + [all_orders[-1][0]])

        for n_order, order in enumerate(all_orders):
            print(f"Training on order {', '.join(order)} | {n_order+1}/{len(all_orders)}")
            print("")

            for n_set in range(len(self.splits[self.dataset])):
                print(f"Training on split {n_set+1}/{len(self.splits[self.dataset])}")
                print("")
                self.reset_model()
                train_data, test_data = self.split_dataset(data, n_set)

                current_train_data = {key: train_data[key] for key in order}
                current_test_data = {key: test_data[key] for key in order}

                test_dfs = self.fit(data=current_train_data, nb_epochs=nb_epochs, test_data=current_test_data)

                if n_set == 0:
                    scores, A = self.compute_cl_scores(test_dfs)
                else:
                    scores_set, A_set = self.compute_cl_scores(test_dfs)
                    scores += scores_set
                    for task in A.keys():
                        A[task] += A_set[task]
            
            scores /= len(self.splits[self.dataset])
            for task in A.keys():
                A[task] /= len(self.splits[self.dataset])

            final_scores.append(scores)

            for task in self.task_type:
                np.save(os.path.join(save_dir, f"A_{task}_{self.protocol}_order{n_order+1}_exp{idx}.npy"), A[task])
            scores.to_csv(os.path.join(save_dir, f"{self.task_type}_{self.protocol}_{self.model_name}_e{nb_epochs}_order{n_order+1}_exp{idx}.csv"))

            print(f"Score for order {', '.join(order)}")
            print("")
            display_scores(scores)
            print("")

        all_index, all_columns = final_scores[0].index, final_scores[0].columns
        aligned_scores = [
            df.reindex(index=all_index, columns=all_columns) for df in final_scores
        ]
        stacked = np.stack([df.values for df in aligned_scores])
        mean_values = np.nanmean(stacked, axis=0)
        mean_scores = pd.DataFrame(mean_values, index=all_index, columns=all_columns)

        print("")
        print("------------------------------------------------------")
        print("")
        print("Final Scores")
        print("")
        display_scores(mean_scores)
        print("")

        mean_scores.to_csv(os.path.join(save_dir, f"{self.task_type}_{self.protocol}_{self.model_name}_global_exp{idx}.csv"))

    def test(self, data, with_mean=True, show=True):
        """
        Test function :
            - Returns the accuracies and f1_scores for each test crisis
        """

        self.model.eval()
        f1_scores = {}
        y_pred, y_true = [[] for _ in range(len(self.task_type))], [[] for _ in range(len(self.task_type))]
        if not show:
            print("")
            print("Model Testing...")
        for task in data.keys():

            if show:
                progress_bar = tqdm(data[task], desc=f"Task : {task}")
            else:
                progress_bar = data[task]
            all_preds = None
            all_labels = None

            for batch in progress_bar:

                x, y = batch
                x, y = [input.to(self.device) for input in x], [target.to(self.device) for target in y]
                with torch.no_grad():
                    outputs = self.model(*x)
                    preds = [output.argmax(dim=1) for output in outputs]

                if all_preds is None:
                    all_preds = [[] for _ in preds]
                    all_labels = [[] for _ in y]

                for i in range(len(preds)):
                    all_preds[i].append(preds[i].detach().cpu())
                    all_labels[i].append(y[i].detach().cpu())

            # Concat all predictions and labels
            f1s = {}
            for i in range(len(all_preds)):
                y_pred[i].append(torch.cat(all_preds[i], dim=0))
                y_true[i].append(torch.cat(all_labels[i], dim=0))

                name = self.task_type[i]
                f1s[name] = f1_score(y_true[i][-1].numpy(), y_pred[i][-1].numpy(), average="macro")

            f1_scores[task] = f1s

        if with_mean:
            f1_mean = {}
            list_columns = f1_scores[list(f1_scores.keys())[0]].keys()
            for col in list_columns:
                f1_mean[col] = np.mean([f1_scores[crisis][col] for crisis in f1_scores.keys()])
            f1_scores["Average"]  = f1_mean

        return self.model_scores(f1_scores)

    def model_scores(self, f1_scores):
        """
        Creates a DataFrame to store the scores of the model
        """
        # On combine les deux dictionnaires
        combined = {}
        for crisis in f1_scores.keys():
            combined[crisis] = {}
            for task_name, f1 in f1_scores[crisis].items():
                combined[crisis][task_name] = f1

        # Création du DataFrame
        df = pd.DataFrame.from_dict(combined, orient="index")
        df = df.sort_index(axis=1)  # trie les colonnes par métrique puis tâche

        return df