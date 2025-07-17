from tqdm import tqdm
import torch
import torch.nn.functional as F
import gc
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import f1_score

from src.utils import display_scores
from models.model import RoBERTaClassifier
from src.approaches.continual import ContinualLearning

class Cumulative(ContinualLearning):

    def __init__(self, data, device):
        super(Cumulative, self).__init__(data)

        self.device = device
        self.nb_classes = data.nb_classes
        self.task_type = data.task_type
        self.model_name = "cumulative"
        self.tokenizer = data.tokenizer
        self.model = RoBERTaClassifier(nb_classes=data.nb_classes, device=device).to(device)
        self.nb_params = sum([p.numel() for p in self.model.parameters()])
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=3e-5)

    def reset_model(self):
        self.model = RoBERTaClassifier(nb_classes=self.nb_classes, device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=3e-5)

    def criterion(self, outputs, targets):
        losses = [ F.cross_entropy(outputs[i], targets[i], reduction="sum") for i in range(len(outputs)) ]
        return sum(losses) / len(losses)

    def fit(self, data, nb_epochs, test_data=None):
        """
        Input :
        - data : sequence of DataLoader which contain the data from each dataset
        Output :
        - None (Changes the model's weights)
        """
        losses = []
        test_scores = []
        n = len(data)
        for j, crisis in enumerate(data.keys()):
            print('')
            print(f"New data {crisis}")
            print('')
            self.model.train()
            self.reset_model()
            loss = None
            if j == 0:
                current_data = data[crisis]
            else:
                new_dataset = ConcatDataset([current_data.dataset, data[crisis].dataset])
                current_data = DataLoader(new_dataset, batch_size=current_data.batch_size, shuffle=True)

            for epoch in range(nb_epochs):
                total_f1_scores = {
                task: 0.0 for task in self.task_type
                }
                progress_bar = tqdm(current_data, desc=f"{crisis} | {j+1}/{n} | Epoch {epoch+1}")

                for i, batch in enumerate(progress_bar):

                    self.optimizer.zero_grad()
                    x, y = batch
                    x, y = [input.to(self.device) for input in x], [target.to(self.device) for target in y]
                    outputs = self.model(*x)
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 3.0)
                    self.optimizer.step()
                    
                    for task in total_f1_scores.keys():
                        total_f1_scores[task] += f1_score(y[self.task_type.index(task)].detach().cpu().numpy(), outputs[self.task_type.index(task)].argmax(dim=1).detach().cpu().numpy(), average="macro")
                    progress_bar.set_postfix({key: value / (i+1) for key, value in total_f1_scores.items()})

                    del x, y, outputs, loss
                    torch.cuda.empty_cache()
                    gc.collect()

            if test_data is not None:
                current_scores = self.test(test_data, show=False)
                test_scores.append(current_scores)
                print("")
                print(f"Scores during training on {crisis}")
                display_scores(current_scores)
                print("")

        return test_scores