from tqdm import tqdm
import torch
import torch.nn.functional as F
import gc
from sklearn.metrics import f1_score

from src.utils import display_scores, EpisodicMemory, stratified_dataloader_reduction
from models.model import RoBERTaClassifier
from src.approaches.continual import ContinualLearning

class NaiveExperienceReplay(ContinualLearning):
    def __init__(self, data, device, M):
        super(NaiveExperienceReplay, self).__init__(data)

        self.device = device
        self.nb_classes = data.nb_classes
        self.model_name = "ner"
        self.M = M
        self.tokenizer = data.tokenizer
        self.task_type = data.task_type
        self.domains = list(data.data.keys())

        self.memory = EpisodicMemory(self.domains)
        self.model = RoBERTaClassifier(nb_classes=self.nb_classes, device=self.device).to(device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=3e-5)

    def reset_model(self):
        self.model = RoBERTaClassifier(nb_classes=self.nb_classes, device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=3e-5)
        self.memory = EpisodicMemory(self.domains)

    def criterion(self, outputs, targets):
        losses = [ F.cross_entropy(outputs[i], targets[i], reduction="sum") for i in range(len(outputs)) ]
        return sum(losses) / len(losses)

    def fit(self, data, nb_epochs, test_data=None):
        self.model.train()
        test_scores = []
        old_crisis = []
        n = len(data.keys())

        for t, crisis in enumerate(data.keys()):
            print(f"\nNew data  {crisis}\nMemory Size : {self.memory.length()}\n")
            first = (t == 0)

            for epoch in range(nb_epochs):
                total_f1_scores = {
                task: 0.0 for task in self.task_type
                }
                progress_bar = tqdm(data[crisis], desc=f"{crisis} | {t+1}/{n} | Epoch {epoch+1}")

                for i, batch in enumerate(progress_bar):
                    self.optimizer.zero_grad()

                    x, y = batch
                    x, y = [input.to(self.device) for input in x], [target.to(self.device) for target in y]
                    outputs = self.model(*x)

                    loss = self.criterion(outputs, y)

                    if not first:
                        for old_c in old_crisis:
                            x_ref, y_ref = self.memory.sample(old_c, batch_size=64)
                            x_ref, y_ref = [xx.to(self.device) for xx in x_ref], [yy.to(self.device) for yy in y_ref]
                            ref_outputs = self.model(*x_ref)
                            loss += self.criterion(ref_outputs, y_ref)
                        loss /= (len(old_crisis) + 1)

                    loss.backward()
                    self.optimizer.step()
                    for task in total_f1_scores.keys():
                        total_f1_scores[task] += f1_score(y[self.task_type.index(task)].detach().cpu().numpy(), outputs[self.task_type.index(task)].argmax(dim=1).detach().cpu().numpy(), average="macro")
                    postfix = {key: value / (i+1) for key, value in total_f1_scores.items()}
                    progress_bar.set_postfix(postfix)

                    del x, y, outputs, loss
                    torch.cuda.empty_cache()
                    gc.collect()

            crisis_to_add = crisis.split("_")[0]
            dataloader_to_add = stratified_dataloader_reduction(data[crisis], total_samples=self.M)
            print("")
            progress_bar = tqdm(dataloader_to_add, desc=f"Adding {crisis_to_add} in memory...")

            for batch in progress_bar:
                x, y = batch
                self.memory.add(x, y, crisis_to_add)
            if crisis_to_add not in old_crisis:
                old_crisis.append(crisis_to_add)
            print("")

            if test_data is not None:
                current_scores = self.test(test_data, show=False)
                test_scores.append(current_scores)

                print(f"\nScores during training on {crisis}")
                print("")
                display_scores(current_scores)

        return test_scores