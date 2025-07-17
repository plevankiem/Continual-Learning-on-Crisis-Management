import matplotlib
import pandas as pd
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, ConcatDataset
from IPython.display import display
from collections import deque


def stratified_dataloader_reduction(dataloader, total_samples=2000):
    """
    Réduit la taille du DataLoader en garantissant l'équilibre des classes.
    Si impossible, essaye sur d'autres tâches, puis sans stratification.
    """
    full_dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    num_tasks = 3  # nombre total de tâches / indices

    for task_idx in range(num_tasks):
        try:
            # Récupérer les labels pour la task task_idx
            all_labels = [full_dataset[i][1][task_idx].item() for i in range(len(full_dataset))]

            # Obtenir indices stratifiés
            selected_indices, _ = train_test_split(
                range(len(full_dataset)),
                train_size=total_samples,
                stratify=all_labels,
                random_state=42
            )
            break 

        except ValueError:
            selected_indices = None

    # Si toutes les stratifications échouent, échantillon aléatoire
    if selected_indices is None:
        selected_indices = random.sample(range(len(full_dataset)), total_samples)

    # Sous-dataset et DataLoader réduit
    reduced_dataset = Subset(full_dataset, selected_indices)
    reduced_dataloader = DataLoader(reduced_dataset, batch_size=batch_size, shuffle=True)

    return reduced_dataloader

def concatenate_events(events):
    """
    Concatenates the datasets from the DataLoader events into a single DataLoader.

    Parameters:
        events (list of DataLoader): List of PyTorch DataLoader instances.

    Returns:
        DataLoader: A new DataLoader over the concatenated dataset, using the batch size of the first DataLoader.
    """
    if not events:
        raise ValueError("The events list is empty.")

    datasets = [loader.dataset for loader in events]
    concatenated_dataset = ConcatDataset(datasets)

    # Récupérer le batch_size du premier DataLoader
    first_loader = events[0]
    batch_size = first_loader.batch_size if first_loader.batch_size is not None else 1  # fallback

    return DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True)

def create_table_styles_for_column_separators(df):
    """
    Crée des styles CSS avec des bordures verticales entre les groupes
    de colonnes (task et metric) dans un DataFrame à 3 niveaux de colonnes.
    """
    styles = []
    prev_task = None
    prev_metric = None

    for i, col in enumerate(df.columns):
        metric, task, model = col

        if i == 0:
            border_left = "3px double white"
            styles.append({
                'selector': f'th.col{i}',
                'props': [('border-left', border_left)]
            })

        border_left = ""
        if prev_task is not None and task != prev_task:
            border_left = "1px solid white"

        if prev_metric is not None and metric != prev_metric:
            border_left = "3px double white"

        if border_left:
            styles.append({
                'selector': f'th.col{i}',
                'props': [('border-left', border_left)]
            })

        prev_task = task
        prev_metric = metric

    return styles

def display_scores(df, final_scores=False):
    """
    Affiche joliment un DataFrame de scores avec colonnes (metric, task, model)
    - Précision à 3 décimales
    - Gradient de fond pour lisibilité
    - Mise en gras du meilleur modèle pour chaque (metric, task) à chaque ligne
    """

    if final_scores:

      if df.columns.nlevels == 3:
        df = df.copy()
        df.columns = df.columns.reorder_levels(['metric', 'task', 'model'])
        df = df.sort_index(axis=1)
      else:
        raise ValueError("The DataFrame should have 3 levels of columns")

      # Init DataFrame de styles vide
      styles = pd.DataFrame("", index=df.index, columns=df.columns)

      # Récupère toutes les paires (metric, task)
      unique_metric_task = df.columns.droplevel("model").unique()

      # Pour chaque couple (metric, task), on trouve les max par ligne
      for (metric, task) in unique_metric_task:
          # Sélection des colonnes correspondant à ce couple
          cols = [col for col in df.columns if col[0] == metric and col[1] == task]

          # Sous-DataFrame des modèles pour ce (metric, task)
          sub_df = df[cols]

          # Localisation du max par ligne
          max_mask = sub_df.eq(sub_df.max(axis=1), axis=0)

          # Application du style en gras
          for col in cols:
              styles[col] = max_mask[col].map(lambda x: "text-decoration: underline" if x else "")

      # Construction du Styler
      styled = (
          df.style
          .format(precision=4)
          .background_gradient(axis=1, cmap="gist_yarg")
          .apply(lambda _: styles, axis=None)
          .set_table_styles(create_table_styles_for_column_separators(df), overwrite=False)
      )

    else:
      styled = (
          df.style
          .format(precision=4)
          .background_gradient(axis=1, cmap="gist_yarg")
      )

    display(styled)

class EpisodicMemory:
    def __init__(self, domains, capacity=None):
        self.buffers = {}
        self.domains = domains
        self.capacity = capacity
        for domain in domains:
            if capacity:
                self.buffers[domain] = deque(maxlen=capacity)
            else:
                self.buffers[domain] = deque()

    def add(self, x, y, domain):
        """Add a new sample to the buffer."""
        for i in range(x[0].size(0)):
            self.buffers[domain].append(([input[i:i+1].detach().cpu() for input in x], [target[i:i+1].detach().cpu() for target in y]))

    def sample(self, domain, batch_size=None):
        """Randomly sample a batch of (x, y) tensors."""
        if batch_size is None:
            real_batch_size = len(self.buffers[domain])
        else:
            real_batch_size = min(batch_size, len(self.buffers[domain]))
        batch = random.sample(self.buffers[domain], real_batch_size)
        list_x_s, list_y_s = zip(*batch)
        nb_inputs, nb_targets = len(list_x_s[0]), len(list_y_s[0])
        x_s = [torch.cat([  list_x_s[i_batch][i_input] for i_batch in range(real_batch_size) ], dim=0) for i_input in range(nb_inputs)]
        y_s = [torch.cat([  list_y_s[i_batch][i_target] for i_batch in range(real_batch_size) ], dim=0) for i_target in range(nb_targets)]

        return x_s, y_s

    def sample_global(self, batch_size):

        all_buffers = []
        for domain in self.domains:
            all_buffers.extend(list(self.buffers[domain]))
        if batch_size is None:
            real_batch_size = len(all_buffers)
        else:
            real_batch_size = min(batch_size, len(all_buffers))

        list_x_s, list_y_s = zip(*random.sample(all_buffers, batch_size))
        nb_inputs, nb_targets = len(list_x_s[0]), len(list_y_s[0])
        x_s = [torch.cat([  list_x_s[i_batch][i_input] for i_batch in range(real_batch_size) ], dim=0) for i_input in range(nb_inputs)]
        y_s = [torch.cat([  list_y_s[i_batch][i_target] for i_batch in range(real_batch_size) ], dim=0) for i_target in range(nb_targets)]

        return x_s, y_s

    def length(self):
        lens = {domain: len(self.buffers[domain]) for domain in self.domains}
        return lens