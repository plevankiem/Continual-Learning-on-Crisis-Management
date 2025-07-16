from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, ConcatDataset

def stratified_dataloader_reduction(dataloader, total_samples=2000):
    # Récupérer tout le dataset à partir du dataloader
    full_dataset = dataloader.dataset
    batch_size = dataloader.batch_size

    # Récupérer tous les labels sous forme de liste
    all_labels = [full_dataset[i][1][0].item() for i in range(len(full_dataset))]  # y est [label_tensor]

    # Obtenir des indices stratifiés
    selected_indices, _ = train_test_split(
        range(len(full_dataset)),
        train_size=total_samples,
        stratify=all_labels,
        random_state=42
    )

    # Créer un sous-dataset
    reduced_dataset = Subset(full_dataset, selected_indices)

    # Nouveau dataloader
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