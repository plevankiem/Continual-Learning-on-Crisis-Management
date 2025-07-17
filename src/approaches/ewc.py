from tqdm import tqdm
import torch
import torch.nn.functional as F
import gc
from sklearn.metrics import f1_score

from src.utils import display_scores
from models.model import RoBERTaClassifier
from src.approaches.continual import ContinualLearning

class ElasticWeightsConsolidation(ContinualLearning):
    """
    Compressed version of EWC as we don't store all the set of parameters and all the Fisher matrix. It is commonly used when working with a large model like BERT.
    """
    def __init__(self, data, device, lambda_):
        super(ElasticWeightsConsolidation, self).__init__(data)

        self.device = device
        self.nb_classes = data.nb_classes
        self.task_type = data.task_type
        self.model_name = "ewc"
        self.tokenizer = data.tokenizer
        self.model = self.load_model(self.dataset, data.nb_classes, device)
        self.nb_params = sum([p.numel() for p in self.model.parameters()])
        self.fisher_matrix = [torch.zeros_like(p) for p in self.model.parameters()]
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=3e-5)
        self.lambda_ = lambda_

    def reset_model(self):
        self.model = self.load_model(self.dataset, self.nb_classes, self.device)
        self.fisher_matrix = [torch.zeros_like(p) for p in self.model.parameters()]
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=3e-5)

    def reg_loss(self, old_params, first):
        """
        Compute the regularization term
        """
        reg_loss = 0.0
        if not first:
            for n, (p_new, p_old) in enumerate(zip(self.model.parameters(), old_params)):
                reg_loss += torch.sum(self.fisher_matrix[n] * (p_new - p_old)**2)
        return reg_loss

    def simple_criterion(self, outputs, targets):
        losses = [ F.cross_entropy(outputs[i], targets[i], reduction="sum") for i in range(len(outputs)) ]
        return sum(losses) / len(losses)

    def compute_fischer(self, dataset):
        """
        Compute the Fischer Matrix F for one dataset
        """
        self.model.eval()
        none_grad = 0

        fisher_matrix = [torch.zeros_like(p) for p in self.model.parameters()]

        progress_bar = tqdm(dataset, desc="    Computing the Fisher Matrix")
        for batch in progress_bar:

            # Reset the gradients
            self.model.zero_grad()

            # Compute the gradients
            x, y = batch
            x, y = [input.to(self.device) for input in x], [target.to(self.device) for target in y]
            outputs = self.model(*x)
            loss = self.simple_criterion(outputs, y)
            loss.backward()

            # Update the matrix
            batch_size = x[0].size(0)
            for n, p in enumerate(self.model.parameters()):
                if p.grad is not None:
                    fisher_matrix[n] += batch_size * (p.grad.detach() ** 2) / len(dataset)
                else:
                    none_grad += 1

            # Saving GPU memory
            del x, y, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        return fisher_matrix

    def update_fisher(self, dataset):
        """Update the fisher matrix with the new dataset"""
        new_fisher_matrix = self.compute_fischer(dataset)
        for n in range(len(self.fisher_matrix)):
            self.fisher_matrix[n] += new_fisher_matrix[n]

    def fit(self, data, nb_epochs, test_data=None):
        """
        Input :
        - data : sequence of DataLoader which contain the data from each dataset
        Output :
        - None (Changes the model's weights)
        """
        self.model.train()
        losses = []
        test_scores = []
        n = len(data.keys())
        for t, crisis in enumerate(data.keys()):
            print('')
            print(f"New data {crisis}")
            print('')

            self.model.train()
            loss = None
            old_params = [p.detach().clone() for p in self.model.parameters()]
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
                    simple_loss = self.simple_criterion(outputs, y)
                    reg_loss = self.reg_loss(old_params, first)
                    loss = simple_loss + self.lambda_ * reg_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                    self.optimizer.step()

                    for task in total_f1_scores.keys():
                        total_f1_scores[task] += f1_score(y[self.task_type.index(task)].detach().cpu().numpy(), outputs[self.task_type.index(task)].argmax(dim=1).detach().cpu().numpy(), average="macro")
                    postfix = {key: value / (i+1) for key, value in total_f1_scores.items()}
                    postfix.update({"Task Loss": simple_loss.item(), "Fisher Loss": reg_loss.item() if not first else 0})
                    progress_bar.set_postfix(postfix)

                    del x, y, outputs, loss
                    torch.cuda.empty_cache()
                    gc.collect()

            print("")
            print("Updating Fisher Matrix...")
            self.update_fisher(data[crisis])
            print("")

            if test_data is not None:
                current_scores = self.test(test_data, show=False)
                test_scores.append(current_scores)
                print("")
                print(f"Scores during training on {crisis}")
                display_scores(current_scores)
                print("")

        return test_scores