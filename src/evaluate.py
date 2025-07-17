import sys
import os
import torch
path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(path, '..')))

from data.scripts.dataset import CrisisDataset
from src.approaches.vanilla import Vanilla

from transformers import RobertaTokenizer

batch_size = 64
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
task_type = ["humanitarian", "urgency", "utility"]

data = CrisisDataset(
    tokenizer=tokenizer,
    batch_size=batch_size,
    task_type=task_type,
    dataset="HumAid"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vanilla = Vanilla(data, device)
vanilla.pipeline(data, nb_epochs=1, idx=1)