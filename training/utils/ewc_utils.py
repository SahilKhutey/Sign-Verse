import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):
        self.model = model
        self.dataset = dataset
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            p.data.zero_()
            precision_matrices[n] = p.clone().detach()

        self.model.eval()
        dataloader = DataLoader(self.dataset, batch_size=1)
        for input, label in dataloader:
            self.model.zero_grad()
            output = self.model(input).view(1, -1)
            label = label.view(1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

def ewc_train(model, optimizer, data_loader, ewc, importance):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        
        # Add EWC penalty
        ewc_loss = ewc.penalty(model)
        total_loss = loss + importance * ewc_loss
        
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
    return epoch_loss / len(data_loader)
