from tqdm import tqdm
import torch
import torch.nn.functional as F
import gc
from sklearn.metrics import f1_score

from src.utils import display_scores
from models.model import RoBERTaClassifier
from src.approaches.continual import ContinualLearning

class SynapticIntelligence(ContinualLearning):
    """
        Implementation of SynapticIntelligence
    """
    def __init__(self, data, device, lambda_):
        super(SynapticIntelligence, self).__init__(data)

        self.device = device
        self.nb_classes = data.nb_classes
        self.task_type = data.task_type
        self.model_name = "si"
        self.tokenizer = data.tokenizer
        self.model = self.load_model(self.dataset, self.nb_classes, self.device)
        self.nb_params = sum([p.numel() for p in self.model.parameters()])
        self.weights = [torch.zeros_like(p) for p in self.model.parameters()]
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=3e-5)
        self.lambda_ = lambda_
        self.ksi = 1e-8

    def reset_model(self):
        self.model = self.load_model(self.dataset, self.nb_classes, self.device)
        self.weights = [torch.zeros_like(p) for p in self.model.parameters()]
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=3e-5)

    def reg_loss(self, old_params, first):
        """
        Compute the regularization term
        """
        reg_loss = 0.0
        if not first:
            for n, (p_new, p_old) in enumerate(zip(self.model.parameters(), old_params)):
                reg_loss += torch.sum(self.weights[n] * (p_new - p_old)**2)
        return reg_loss

    def simple_criterion(self, outputs, targets):
        losses = [ F.cross_entropy(outputs[i], targets[i], reduction="sum") for i in range(len(outputs)) ]
        return sum(losses) / len(losses)

    def cumulative_trajectory(self, current_weights, grad, new_params, prev_params):
        """
        Update the Cumulative Weights
        """
        for n in range(len(self.weights)):
            current_weights[n] -= grad[n] * ( new_params[n] - prev_params[n] )

        return current_weights

    @torch.no_grad()                       # ① coupe Autograd
    def update_weights(self, current_weights, params, old_params):
        params   = list(params)            # ② consomme le générateur UNE fois
        old_params = list(old_params)
        for n, (p_new, p_old) in enumerate(zip(params, old_params)):
            delta_omega = current_weights[n] / ((p_new - p_old)**2 + self.ksi)
            self.weights[n].add_(delta_omega)   # ③ add_ évite une nouvelle allocation

    def fit(self, data, nb_epochs, test_data=None):
        """
        Input :
        - data : sequence of DataLoader which contain the data from each dataset
        Output :
        - None (Changes the model's weights)
        """
        self.model.train()
        test_scores = []
        n = len(data.keys())
        for t, crisis in enumerate(data.keys()):
            print('')
            print(f"New data {crisis}")
            print('')

            current_weights = [torch.zeros_like(p) for p in self.model.parameters()]
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

                    # Inputs for the current batch
                    self.optimizer.zero_grad()
                    x, y = batch
                    x, y = [input.to(self.device) for input in x], [target.to(self.device) for target in y]

                    # ====== Forward ======
                    outputs = self.model(*x)
                    simple_loss = self.simple_criterion(outputs, y)
                    reg_loss    = self.reg_loss(old_params, first)   # 0 si first == True
                    loss        = simple_loss + self.lambda_ * reg_loss

                    # ====== Backward + step ======
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)

                    # Copie des gradients AVANT .step() (sinon ils seront remis à 0)
                    grads = []
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grads.append(p.grad.detach().clone())
                        else:
                            grads.append(torch.zeros_like(p))
                    params0 = [p.detach().clone()      for p in self.model.parameters()]

                    self.optimizer.step()
                    params1 = [p.detach().clone() for p in self.model.parameters()]

                    # Update the trajectory weights
                    current_weights = self.cumulative_trajectory(current_weights, grads, params1, params0)

                    for task in total_f1_scores.keys():
                        total_f1_scores[task] += f1_score(y[self.task_type.index(task)].detach().cpu().numpy(), outputs[self.task_type.index(task)].argmax(dim=1).detach().cpu().numpy(), average="macro")
                    postfix = {key: value / (i+1) for key, value in total_f1_scores.items()}
                    postfix.update({"Task Loss": simple_loss.item(), "Regularization Loss": reg_loss.item() if not first else 0})
                    progress_bar.set_postfix(postfix)

                    del x, y, outputs, loss
                    torch.cuda.empty_cache()
                    gc.collect()

            print("")
            print("Updating Synaptic Weights...")
            self.update_weights(current_weights, self.model.parameters(), old_params)
            print("")
            del current_weights, grads, params0, params1
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